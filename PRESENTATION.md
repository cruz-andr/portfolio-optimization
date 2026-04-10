# Presentation Script: Portfolio Optimization with Deep Reinforcement Learning

A four-person walkthrough of the project, roughly 16–20 minutes total. Each section is written as what the presenter actually says, with notes on what to show on screen in *italics*.

**Presenters:**
1. Aryan Kalaskar — Motivation and problem setup
2. Manuel Torres — Data pipeline and trading environment
3. Crosby Sayan — The DQN agent and extensions (HMM regimes, factors)
4. Andres Cruz — Evaluation, results, and takeaways

---

## Part 1 — Aryan Kalaskar: Motivation and Problem Setup *(~4 min)*

*Slide: title slide with the four authors.*

Good afternoon, everyone. I'm Aryan, and today my teammates Manuel, Crosby, Andres, and I are going to walk you through our project: **Portfolio Optimization with Deep Reinforcement Learning**.

Here is the core question we set out to answer: **can a reinforcement learning agent learn to trade a basket of stocks better than the standard benchmarks you'd see on a finance textbook?** Specifically, better than just buying and holding the market, or following simple momentum and mean-reversion rules.

*Slide: one-line problem statement — "hold cash, or be invested?"*

The setup is actually really simple to describe. Every trading day, for every stock in our universe, our agent looks at a snapshot of that stock — things like moving averages, volatility, fundamentals — and it decides exactly one thing: **hold cash, or be invested**. At the end of each day we equal-weight whatever the agent chose to hold, pay a small transaction cost on any new positions, and compound the returns forward. That gives us a portfolio return curve we can compare head-to-head against the benchmarks.

*Slide: title page of the Pigorsch & Schäfer 2021 paper.*

Our starting point is a 2021 paper by Pigorsch and Schäfer called *High-Dimensional Stock Portfolio Trading with Deep Reinforcement Learning*. Stage one of our project was a faithful reproduction of their core Deep Q-Network approach. Stage two was extending it in three directions of our own:

1. **A Sharpe-ratio bonus in the reward function**, so the agent isn't just chasing raw returns but is penalized for volatility.
2. **A Hidden Markov Model for market-regime awareness**, so the agent knows whether we're in a bull, bear, or high-volatility state.
3. **Fama-French-style factor features**, so the agent has access to cross-sectional information about size, value, momentum, and quality.

Now, one question you might be asking is: **why reinforcement learning at all?** Why not just use a regression or a classifier?

The honest answer is that traditional tabular Q-learning breaks down on real markets for two reasons. First, the state space is continuous and high-dimensional — you can't just put it in a lookup table. Second, new stocks get issued all the time, so not every ticker has a full price history, and classical methods struggle with that. DQN handles both problems naturally: it generalizes across stocks through the neural network, and it can make decisions under partial observability the same way the original Atari DQN did.

With that context, I'm going to hand things over to Manuel, who'll walk you through how we actually turned raw market data into something an RL agent can learn from.

---

## Part 2 — Manuel Torres: Data Pipeline and Trading Environment *(~4 min)*

Thanks, Aryan. So I'm Manuel, and I'm going to cover two things: the data pipeline that feeds the agent, and the custom trading environment that the agent actually lives inside.

*Slide: data pipeline diagram — parquet → filter → subset → split → normalize.*

We started with a parquet file of US equity data covering 2010 through mid-2021. That's about 6.4 million rows and over 3,200 unique stocks. Obviously we can't and don't want to train on everything at once, so the pipeline does five things in order.

First, it drops any rows with missing features — we didn't want to deal with imputation noise for a proof of concept. Second, it keeps only stocks with at least 250 trading days of history, so the agent always has a reasonable window to look back on. Third — and this is important — we take a **50-stock proof-of-concept subset**: the 20 largest stocks by market cap, the 20 smallest, and 10 random mid-caps. That gives us cap diversity without blowing up training time. Fourth, we split chronologically: training is everything before 2019, validation is 2019, and test is 2020 onward. Fifth, we z-score every feature using **training set statistics only** — no look-ahead leakage.

*Slide: bulleted list of the 28 features.*

The agent's state on any given day is 28 features — or 32 if we turn on factors. Breaking it down: 17 technical features like simple and exponential moving averages at 5, 10, 20, 50, 100, and 200-day windows, plus rolling standard deviations. 10 fundamentals — things like ROE, ROA, current ratio, book-to-market, debt ratio. One price feature, the close. Optionally 4 factor ranks. Then we also append the agent's current position and the current market regime.

*Slide: environment MDP diagram with action, state, reward.*

Now, the environment itself. This follows the paper's per-asset MDP formulation. What that means is each training episode focuses on **one stock at a time**, chosen at random from the training universe, and walks forward day by day.

The action space is binary: zero means hold cash, one means invested. The state is the feature vector plus position plus regime. The reward structure is where it gets interesting:

- **When invested:** the agent gets the stock's next-day return, minus a transaction cost if it just opened this position.
- **When in cash:** the agent gets the **cross-sectional mean return across all stocks that day.**

That second piece is the paper's clever trick. If sitting in cash always gave zero reward, the agent would learn to be lazy and do nothing. By setting the cash reward to the market average, we're effectively punishing the agent for sitting out a rally — "doing nothing" now has real opportunity cost.

*Slide: table of the three reward variants.*

On top of that base reward, we built three variants. `base` is the pure paper reward. `sharpe` adds a bonus proportional to the stock's rolling 20-day Sharpe ratio, so the agent cares about risk-adjusted return. And `sharpe+regime` takes that bonus and scales it by a regime weight, so the agent is pushed to take more risk when the market is in a bear state. Crosby is going to talk about where that regime signal comes from in a moment.

Over to you, Crosby.

---

## Part 3 — Crosby Sayan: The DQN Agent and Extensions *(~4 min)*

Thanks, Manuel. So my job is to walk you through the brain of the project — the DQN agent itself — and then the two extensions I worked on most closely, which are the HMM regime model and the factor model.

*Slide: neural network architecture diagram.*

Let's start with the agent. The network is actually pretty modest: a **two-hidden-layer ReLU MLP** with an output dimension of two — one Q-value per action. Hidden width is a hyperparameter, and we use three values: 32, 64, and 128. I'll come back to why three in a minute.

For the training loop we followed the paper closely but scaled things down because we're working with laptop-level compute. We run 500,000 environment steps per agent — the paper uses 3 million — with an epsilon-greedy policy at epsilon 0.3. We do one gradient update every 20 environment steps, with batch size 1,024. The replay buffer holds 50,000 transitions, the target network syncs every 1,000 steps for stability, and we use Adam with a learning rate of 5e-4 and gradient clipping at norm 1.0.

*Slide: validation checkpointing diagram.*

One thing we were careful to replicate is **validation checkpointing**. Every 10,000 training steps we pause, run the current network on the 2019 validation year, and keep whichever weights produced the best cumulative return. That matches Algorithm 1 in the paper and it's really important — without it the agent often overfits late in training.

Now, the reason I mentioned three hidden widths earlier: we actually train **three independent networks per configuration**, one at each width, and at test time we ensemble them by majority voting. A stock gets into the portfolio only if at least two of the three networks agree to invest in it.

*Slide: HMM state diagram with three regimes.*

Okay, extension one: the **Hidden Markov Model for regimes**. We fit a 3-state Gaussian HMM from scratch in PyTorch — forward-backward, EM, and Viterbi are all hand-rolled, no `hmmlearn` dependency. The observations are daily cross-sectional mean return and cross-sectional volatility.

After training, the three states look interpretable: one is basically a **bear** regime with negative mean returns, one is a **high-volatility** regime with positive mean but large swings, and one is a **calm** regime with slightly positive mean and low volatility. Every date in every split gets a hard regime label and soft probabilities, which the environment uses to scale the Sharpe bonus.

*Slide: four factor definitions.*

Extension two: **Fama-French-style factors**. Four cross-sectional percentile ranks, recomputed every trading day:

- **Size**, ranking on market cap.
- **Value**, ranking on book-to-market.
- **Momentum**, ranking on the ratio of the 5-day to 200-day moving average.
- **Quality**, ranking on the average of ROE and ROA.

These are deliberately simple compared to commercial factor models like Barra or Axioma — we wanted to see how much signal you could get from the most basic possible factor set.

With all of that in place, the question becomes: **does any of it actually work?** And for that, I'm handing over to Andres.

---

## Part 4 — Andres Cruz: Evaluation, Results, and Takeaways *(~4 min)*

Thanks, Crosby. So my job is to tell you whether all of this actually beat the benchmarks — and as a small spoiler, the answer is interesting.

*Slide: evaluation protocol diagram.*

First, how we evaluate. On the test set, which starts in January 2020, we do the following for every trading day: each of the three ensemble networks votes on every stock in the universe. A stock is included in the portfolio if at least two out of three networks vote to invest. We then equal-weight the selected stocks, pay transaction costs on any new positions, and record the portfolio return.

One implementation detail worth calling out: if the ensemble picks fewer than three stocks on a given day, we fall back to holding the full buy-and-hold basket. That's to avoid unrealistic concentration on days where the agent is indecisive.

We benchmark against three strategies from the paper: plain **buy and hold**, a **5-day momentum** rule, and a **5-day reversion** rule. We report cumulative return, annualized Sharpe ratio, maximum drawdown, and win rate against buy-and-hold.

*Slide: big results table.*

Now the results. Just to set expectations — our test window **includes the COVID crash and recovery**, which is about as stressful as test windows get.

Buy and hold over that period returned about **+114 percent** with a Sharpe of 1.65 and a max drawdown of about negative 32 percent. Momentum actually **lost money** — it came in around negative 21 percent. Reversion did well at **+141 percent**.

Now the DQN results. The base reward agent beat buy-and-hold decisively at **+189 percent**. But the real standout is the **Sharpe reward agent with no factors and no regime**. That one came in at **+303 percent cumulative return**, a Sharpe ratio of **2.27**, a max drawdown of only **negative 21 percent**, and a **52.8 percent win rate** against buy-and-hold on a day-by-day basis.

To put that in perspective: that's roughly **2.7 times the return of buy-and-hold with a shallower drawdown.** On a risk-adjusted basis it's the best thing in the table by a clear margin.

*Slide: equity curve plot comparing Sharpe agent vs benchmarks.*

Here's the equity curve. You can see the agent tracking the benchmarks for a while, then pulling meaningfully ahead through the recovery from the COVID crash.

*Slide: the "surprising" finding.*

Now here's the honest and interesting part. We expected the **sharpe+regime** agent and the **factors** agent to do even better, because we were giving the agent more information. But on this particular test window they actually **underperformed the pure Sharpe agent**. The regime version still had a respectable cumulative return of 296 percent but a much worse drawdown, and the factors version dropped to 225 percent.

We think there are a few possible explanations. One, the COVID regime shift might have been too abrupt for the HMM to label cleanly in the test window. Two, adding features without adding training compute probably hurt sample efficiency — more inputs, same 500k steps. Three, our factor construction is simple, and a simple factor signal in a weird macro year might just be noise.

*Slide: milestones and next steps.*

That leads into what's next. The three things we want to investigate are:

1. **Out-of-sample stress testing** — run the same pipeline through 2008 so we have two crisis windows to compare.
2. **More training compute on the richer configurations** — see if the regime and factors agents were just undercooked.
3. **A proper ablation** on the reward bonus to figure out how much of the Sharpe agent's outperformance is signal versus lucky training seed.

*Slide: thank you / questions.*

And that's our project. The short version is: a straightforward DQN, trained with a Sharpe-aware reward, **beat buy-and-hold, momentum, and reversion through the COVID crash on a risk-adjusted basis**. The extensions we expected to help the most were more nuanced than we hoped, and understanding why is the next step.

Thank you — we'd love to take your questions.

---

## Appendix: Timing and Q&A Notes

- Total target runtime: 16–20 minutes for the script, leaving 5–10 minutes for Q&A.
- If time is tight, the most trimmable section is Part 3 — Crosby can skip the detailed training hyperparameters and just name-drop DQN, target network, replay buffer.
- Likely audience questions and who should field them:
  - *"Why only 50 stocks?"* → Manuel. (Compute budget, cap diversity, proof-of-concept scope.)
  - *"Why DQN and not PPO or SAC?"* → Aryan or Crosby. (Paper reproduction, discrete action space is a natural DQN fit.)
  - *"Isn't 2020 cherry-picked?"* → Andres. (Agree it's a single window; 2008 stress test is explicitly on the roadmap.)
  - *"How do you handle transaction costs realistically?"* → Manuel. (Flat cost per new position, no slippage model — an honest simplification.)
  - *"Are the HMM regimes predictive or coincident?"* → Crosby. (Viterbi is retrospective; in-sample the labels are clean but out-of-sample lag is a real concern.)
