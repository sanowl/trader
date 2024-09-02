import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import ta
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeType(Enum):
    NO_TRADE = 0
    LONG = 1
    SHORT = 2

@dataclass
class TradeInfo:
    type: TradeType    
    entry: float
    tp: float
    sl: float
    rr_ratio: float

class FeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__()
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomNetwork(nn.Module):
    def __init__(self, feature_dim: int, last_layer_dim_pi: int, last_layer_dim_vf: int):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_pi)
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, 
                 lr_schedule: callable, *args: Any, **kwargs: Any):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self._build_mlp_extractor()

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = FeatureExtractor(self.observation_space)
        self.mlp_extractor = CustomNetwork(64, self.action_space.shape[0], 1)

class DeepTraderEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, num_candles: int = 300, render_mode: Optional[str] = None):
        super().__init__()
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(num_candles, int) or num_candles <= 0:
            raise ValueError("num_candles must be a positive integer")

        self.render_mode = render_mode
        self.num_candles = min(num_candles, len(data))
        self.data = self._preprocess_data(data)
        self.lookahead_steps = 200

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15, self.num_candles), dtype=np.float32)

        self.scaler = StandardScaler()
        self.reset()

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Data must contain columns: {required_columns}")

            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband(), ta.volatility.BollingerBands(df['Close']).bollinger_mavg(), ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            df['Stoch_K'], df['Stoch_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch(), ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            raise

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = self.num_candles
        self.current_trade: Optional[TradeInfo] = None
        self.portfolio_value = 10000
        self.trades_executed = 0
        self.cumulative_returns = 0
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not isinstance(action, np.ndarray) or action.shape != (4,):
            raise ValueError("Invalid action shape. Expected (4,)")

        try:
            trade_type = TradeType(int(np.clip(action[0], -1, 1) * 1.5 + 1.5))
            entry_offset, tp_offset, sl_offset = action[1:]

            current_price = self.data.iloc[self.current_step]['Close']
            entry_point = current_price * (1 + entry_offset * 0.01)
            tp = entry_point * (1 + tp_offset * 0.01)
            sl = entry_point * (1 - sl_offset * 0.01)

            self.current_trade = TradeInfo(
                type=trade_type,
                entry=entry_point,
                tp=tp,
                sl=sl,
                rr_ratio=abs(tp - entry_point) / abs(entry_point - sl) if abs(entry_point - sl) > 1e-6 else float('inf')
            )

            reward, done = self._calculate_reward()
            self.current_step += 1
            terminated = done or self.current_step >= len(self.data) - 1
            truncated = False

            if self.render_mode == 'human' and self.current_step % 1000 == 0:
                self.render()

            return self._get_observation(), reward, terminated, truncated, {
                'portfolio_value': self.portfolio_value,
                'trades_executed': self.trades_executed,
                'cumulative_returns': self.cumulative_returns
            }
        except Exception as e:
            logger.error(f"Error in step function: {str(e)}")
            raise

    def _calculate_reward(self) -> Tuple[float, bool]:
        if self.current_trade.type == TradeType.NO_TRADE:
            return -0.01, False  # Small penalty for no trade

        reward = 0
        done = False
        rr_ratio = self.current_trade.rr_ratio

        if not 1.5 <= rr_ratio <= 3:
            return -0.1, True  # Penalize poor risk-reward setups

        for future_step in range(self.current_step, min(self.current_step + self.lookahead_steps, len(self.data))):
            candle = self.data.iloc[future_step]
            if self.current_trade.type == TradeType.LONG:
                if candle['High'] >= self.current_trade.tp:
                    reward = 1.0 + (rr_ratio * 0.1)  # Bonus for good RR ratio
                    self.portfolio_value *= (1 + reward * 0.01)
                    done = True
                    break
                elif candle['Low'] <= self.current_trade.sl:
                    reward = -1.0
                    self.portfolio_value *= (1 + reward * 0.01)
                    done = True
                    break
            elif self.current_trade.type == TradeType.SHORT:
                if candle['Low'] <= self.current_trade.tp:
                    reward = 1.0 + (rr_ratio * 0.1)
                    self.portfolio_value *= (1 + reward * 0.01)
                    done = True
                    break
                elif candle['High'] >= self.current_trade.sl:
                    reward = -1.0
                    self.portfolio_value *= (1 + reward * 0.01)
                    done = True
                    break

        if done:
            self.trades_executed += 1
            self.cumulative_returns += reward

        return reward, done

    def _get_observation(self) -> np.ndarray:
        try:
            start = self.current_step - self.num_candles
            end = self.current_step
            df = self.data.iloc[start:end]

            obs = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_K', 'Stoch_D']].values
            scaled_obs = self.scaler.fit_transform(obs.T)

            time_index = (df.index.astype(int).values - df.index[0].astype(int)) / 1e9
            scaled_time = (time_index - np.mean(time_index)) / np.std(time_index)

            full_obs = np.vstack([scaled_obs, scaled_time.reshape(1, -1)])
            
            if full_obs.shape[0] < 15:
                padding = np.zeros((15 - full_obs.shape[0], full_obs.shape[1]))
                full_obs = np.vstack([full_obs, padding])
            elif full_obs.shape[0] > 15:
                full_obs = full_obs[:15, :]

            return full_obs
        except Exception as e:
            logger.error(f"Error in getting observation: {str(e)}")
            raise

    def render(self) -> None:
        if self.current_trade:
            logger.info(f"\nStep: {self.current_step}")
            logger.info(f"Trade Type: {self.current_trade.type.name}")
            logger.info(f"Entry: {self.current_trade.entry:.5f}")
            logger.info(f"TP: {self.current_trade.tp:.5f}")
            logger.info(f"SL: {self.current_trade.sl:.5f}")
            logger.info(f"R/R Ratio: {self.current_trade.rr_ratio:.2f}")
            logger.info(f"Portfolio Value: {self.portfolio_value:.2f}")
            logger.info(f"Trades Executed: {self.trades_executed}")
            logger.info(f"Cumulative Returns: {self.cumulative_returns:.2f}")

    def visualize_trade(self, lookback: int = 100, lookahead: int = 50) -> None:
        try:
            start = max(0, self.current_step - lookback)
            end = min(len(self.data), self.current_step + lookahead)
            df = self.data.iloc[start:end]

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                row_heights=[0.6, 0.2, 0.2])

            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=3, col=1)

            if self.current_trade:
                fig.add_hline(y=self.current_trade.entry, line_color='blue', line_width=1, line_dash='dash', row=1, col=1)
                fig.add_hline(y=self.current_trade.tp, line_color='green', line_width=1, line_dash='dash', row=1, col=1)
                fig.add_hline(y=self.current_trade.sl, line_color='red', line_width=1, line_dash='dash', row=1, col=1)

            fig.update_layout(height=800, title_text="Trade Visualization")
            fig.show()
        except Exception as e:
            logger.error(f"Error in visualizing trade: {str(e)}")
            raise

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards: List[float] = []

    def _on_step(self) -> bool:
        try:
            if self.locals.get("dones"):
                reward = self.locals.get("rewards")
                if not isinstance(reward, (float, np.floating, int, np.integer)):
                    raise TypeError("Reward must be a numeric type")
                self.rewards.append(reward)
                self.logger.record('episode_reward', np.mean(self.rewards[-100:]))
                self.logger.record('portfolio_value', self.training_env.get_attr('portfolio_value')[0])
                self.logger.record('trades_executed', self.training_env.get_attr('trades_executed')[0])
                self.logger.record('cumulative_returns', self.training_env.get_attr('cumulative_returns')[0])
            return True
        except Exception as e:
            logger.error(f"Error in TensorboardCallback: {str(e)}")
            return False

def create_env(data: pd.DataFrame, render_mode: Optional[str] = None) -> gym.Env:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    try:
        return Monitor(DeepTraderEnv(data, render_mode=render_mode))
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise

def train_model(data: pd.DataFrame, total_timesteps: int = 1000000) -> PPO:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(total_timesteps, int) or total_timesteps <= 0:
        raise ValueError("total_timesteps must be a positive integer")

    try:
        # Create environments for training and evaluation
        env = DummyVecEnv([lambda: create_env(data)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        eval_env = DummyVecEnv([lambda: create_env(data.copy())])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

        # Initialize the PPO model
        model = PPO(
            policy=CustomActorCriticPolicy,
            env=env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            tensorboard_log="./ppo_deeptrader_tensorboard/"
        )

        # Define the evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./best_model/',
            log_path='./logs/',
            eval_freq=10000,
            deterministic=True,
            render=False
        )

        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[TensorboardCallback(), eval_callback]
        )

        return model
    except Exception as e:
        logger.error(f"Error in training model: {str(e)}")
        raise

def backtest(model: PPO, data: pd.DataFrame, initial_balance: float = 10000) -> pd.DataFrame:
    if not isinstance(model, PPO):
        raise TypeError("model must be an instance of PPO")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(initial_balance, (float, int)) or initial_balance <= 0:
        raise ValueError("initial_balance must be a positive number")

    try:
        env = create_env(data)
        obs, _ = env.reset()
        done = False
        balance = initial_balance
        trades = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if env.unwrapped.current_trade and env.unwrapped.current_trade.type != TradeType.NO_TRADE:
                trades.append({
                    'timestamp': data.index[env.unwrapped.current_step],
                    'type': env.unwrapped.current_trade.type.name,
                    'entry': env.unwrapped.current_trade.entry,
                    'tp': env.unwrapped.current_trade.tp,
                    'sl': env.unwrapped.current_trade.sl,
                    'reward': reward,
                    'balance': balance * (1 + reward * 0.01)
                })
                balance = trades[-1]['balance']

        return pd.DataFrame(trades)
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise

def analyze_backtest_results(backtest_results: pd.DataFrame) -> None:
    if not isinstance(backtest_results, pd.DataFrame):
        raise TypeError("backtest_results must be a pandas DataFrame")

    try:
        total_trades = len(backtest_results)
        profitable_trades = len(backtest_results[backtest_results['reward'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        initial_balance = 10000  # Assuming initial balance of 10000
        final_balance = backtest_results['balance'].iloc[-1] if not backtest_results.empty else initial_balance
        total_return = (final_balance - initial_balance) / initial_balance * 100

        logger.info(f"Backtest Analysis:")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Profitable Trades: {profitable_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Initial Balance: ${initial_balance:.2f}")
        logger.info(f"Final Balance: ${final_balance:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
    except Exception as e:
        logger.error(f"Error in analyzing backtest results: {str(e)}")
        raise

def visualize_backtest_results(data: pd.DataFrame, backtest_results: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(backtest_results, pd.DataFrame):
        raise TypeError("backtest_results must be a pandas DataFrame")

    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Price', 'Balance'), row_heights=[0.7, 0.3])

        # Price chart
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)

        # Balance chart
        fig.add_trace(go.Scatter(x=backtest_results['timestamp'], y=backtest_results['balance'], mode='lines', name='Balance'), row=2, col=1)

        # Add trade markers
        for _, trade in backtest_results.iterrows():
            color = 'green' if trade['reward'] > 0 else 'red'
            symbol = 'triangle-up' if trade['type'] == 'LONG' else 'triangle-down'
            fig.add_trace(go.Scatter(x=[trade['timestamp']], y=[trade['entry']], mode='markers', marker=dict(color=color, symbol=symbol, size=10), name=f"{trade['type']} Trade"), row=1, col=1)

        fig.update_layout(height=800, title_text="Backtest Results Visualization")
        fig.show()
    except Exception as e:
        logger.error(f"Error in visualizing backtest results: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        ticker_symbol = "GBPAUD=X"
        data = yf.Ticker(ticker_symbol).history(period="2y", interval="1h")

        if data.empty:
            raise ValueError(f"No data retrieved for ticker symbol: {ticker_symbol}")

        model = train_model(data)
        model.save("ppo_deep_trader_final")

        mean_reward, std_reward = evaluate_policy(model, create_env(data), n_eval_episodes=10)
        logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        backtest_results = backtest(model, data)
        analyze_backtest_results(backtest_results)
        visualize_backtest_results(data, backtest_results)

        logger.info("Training and backtesting completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")
