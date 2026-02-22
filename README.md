# NBA MVP Prediction Model

## How It Works

We built an XGBoost model that predicts MVP vote share from player statistics. The idea is simple: we feed the model 45 features for every qualified player in a season — conventional stats (points, rebounds, assists), advanced metrics (VORP, Win Shares, BPM, PER), and team context (win percentage, total wins, conference seed) — and it predicts what fraction of the maximum possible MVP votes that player would receive.

**XGBoost** works by building 500 small decision trees in sequence. The first tree makes a rough prediction ("is this player's VORP above 6?"), and each subsequent tree focuses specifically on correcting the previous trees' mistakes. The final prediction is the sum of all 500 trees. This lets it capture nonlinear patterns that matter for MVP voting — like how a 30 PPG scorer on a 1-seed gets way more votes than a 30 PPG scorer on an 8-seed.

We trained on every season from 1980 to 2025 and validated using **leave-one-year-out cross-validation**: for each season, we trained on all other seasons and predicted that year's MVP race blind. Predictions are normalized within each year so vote shares sum to 2.6 (matching the actual voting system where each ballot distributes 26 points across 5 players).

## Results Across All 46 Seasons

- **Top-1 accuracy: 65% (30/46)** — correctly predicted the exact MVP
- **Top-3 accuracy: 96% (44/46)** — actual MVP was in predicted top 3
- **Only 2 complete misses**: Steve Nash 2005 and 2006 (narrative-driven picks no stats model would catch)

| Year | Actual MVP | Predicted #1 | Hit? |
|------|-----------|-------------|------|
| 1980 | Kareem Abdul-Jabbar | Kareem Abdul-Jabbar | Y |
| 1981 | Julius Erving | Julius Erving | Y |
| 1982 | Moses Malone | Larry Bird | top 3 |
| 1983 | Moses Malone | Larry Bird | top 3 |
| 1984 | Larry Bird | Larry Bird | Y |
| 1985 | Larry Bird | Larry Bird | Y |
| 1986 | Larry Bird | Larry Bird | Y |
| 1987 | Magic Johnson | Magic Johnson | Y |
| 1988 | Michael Jordan | Larry Bird | top 3 |
| 1989 | Magic Johnson | Michael Jordan | top 3 |
| 1990 | Magic Johnson | Magic Johnson | Y |
| 1991 | Michael Jordan | Michael Jordan | Y |
| 1992 | Michael Jordan | Michael Jordan | Y |
| 1993 | Charles Barkley | Michael Jordan | top 3 |
| 1994 | Hakeem Olajuwon | David Robinson | top 3 |
| 1995 | David Robinson | David Robinson | Y |
| 1996 | Michael Jordan | Michael Jordan | Y |
| 1997 | Karl Malone | Michael Jordan | top 3 |
| 1998 | Michael Jordan | Karl Malone | top 3 |
| 1999 | Tim Duncan | Tim Duncan | Y |
| 2000 | Shaquille O'Neal | Shaquille O'Neal | Y |
| 2001 | Allen Iverson | Shaquille O'Neal | top 3 |
| 2002 | Tim Duncan | Tim Duncan | Y |
| 2003 | Tim Duncan | Tim Duncan | Y |
| 2004 | Kevin Garnett | Kevin Garnett | Y |
| 2005 | Steve Nash | Kevin Garnett | **miss** |
| 2006 | Steve Nash | Dirk Nowitzki | **miss** |
| 2007 | Dirk Nowitzki | Dirk Nowitzki | Y |
| 2008 | Kobe Bryant | LeBron James | top 3 |
| 2009 | LeBron James | LeBron James | Y |
| 2010 | LeBron James | LeBron James | Y |
| 2011 | Derrick Rose | LeBron James | top 3 |
| 2012 | LeBron James | LeBron James | Y |
| 2013 | LeBron James | LeBron James | Y |
| 2014 | Kevin Durant | Kevin Durant | Y |
| 2015 | Stephen Curry | Stephen Curry | Y |
| 2016 | Stephen Curry | Stephen Curry | Y |
| 2017 | Russell Westbrook | Kawhi Leonard | top 3 |
| 2018 | James Harden | James Harden | Y |
| 2019 | Giannis Antetokounmpo | Giannis Antetokounmpo | Y |
| 2020 | Giannis Antetokounmpo | Giannis Antetokounmpo | Y |
| 2021 | Nikola Jokic | Nikola Jokic | Y |
| 2022 | Nikola Jokic | Giannis Antetokounmpo | top 3 |
| 2023 | Joel Embiid | Nikola Jokic | top 3 |
| 2024 | Nikola Jokic | Nikola Jokic | Y |
| 2025 | Shai Gilgeous-Alexander | Shai Gilgeous-Alexander | Y |

## The Limits of Stats: Why 65% Is Actually the Ceiling

The model gets the right guy 65% of the time and has him in the top 3 nearly every year — but it still misses 16 winners. That's not a flaw in the model so much as a reflection of what MVP voting actually is. It's not a pure stats award. When two or three players have similar numbers on winning teams, the deciding factor is often narrative — something no box score can capture.

Look at the pattern in the misses:

- **Voter fatigue:** The model loves Michael Jordan in 1993 and 1997, and for good reason — his stats were the best. But voters gave it to Barkley and Malone partly because Jordan had already won three MVPs and the award felt stale. Same with LeBron in 2011 (Rose won) and 2008 (Kobe won after years of "he deserves one").
- **Story of the season:** Allen Iverson in 2001 dragged a mediocre Sixers roster to the 1-seed. Westbrook in 2017 averaged a triple-double. Derrick Rose was the youngest MVP ever on a 62-win Bulls team. These weren't just stat lines — they were moments that captured voters' imaginations.
- **Nash's revolution:** The two complete misses (2005, 2006) are Steve Nash winning back-to-back despite stats that don't jump off the page. Nash transformed how the Suns played and voters rewarded the innovation. No regression model has a feature for "changed the game."

The takeaway: stats get you 96% of the way to the right answer (the actual MVP is almost always in the model's top 3). But the last mile — picking the winner from a tight group of 2-3 elite candidates — is where human storytelling takes over. MVP voting is part math, part mythology, and a model can only do the math.

## 2025-26 MVP Predictions (as of mid-February)

| Rank | Player | Team | PPG | Wins | Seed | Pred. Share |
|------|--------|------|-----|------|------|-------------|
| 1 | Nikola Jokic | DEN | 28.7 | 35 | W3 | 0.683 |
| 2 | Shai Gilgeous-Alexander | OKC | 31.8 | 42 | W1 | 0.620 |
| 3 | Giannis Antetokounmpo | MIL | 28.0 | 23 | E12 | 0.269 |
| 4 | Luka Doncic | LAL | 32.8 | 33 | W5 | 0.260 |
| 5 | Jaylen Brown | BOS | 29.3 | 35 | E2 | 0.240 |
| 6 | Victor Wembanyama | SAS | 24.4 | 38 | W2 | 0.236 |
| 7 | Cade Cunningham | DET | 25.3 | 40 | E1 | 0.097 |

The model has Jokic as the frontrunner driven by his historically elite advanced numbers (VORP, WS, BPM), with SGA close behind thanks to leading the top-seeded Thunder. Giannis slots in third on raw production despite Milwaukee's poor record dragging him down. Wembanyama is an interesting dark horse — the Spurs sitting at the 2-seed is a huge boost.

Of course, if the model's own track record tells us anything, it's that Jokic vs. SGA is exactly the kind of tight two-man race where narrative will be the tiebreaker.
