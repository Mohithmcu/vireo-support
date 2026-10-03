# Submission Form — Vireo Audio Support Tickets (Set B)

---

## What did you build, and what business outcome does it move? State the number and the money.

A four-tab Streamlit dashboard backed by a pure-pandas pipeline (`vireo/pipeline.py`) and an independent statistical validation script (`validate.py`).

**Tab 1 — Per-agent (raw).** Ranks all 38 Tier 1 agents by raw CSAT and median handle time. Bottom 10 highlighted. An explicit alert explains why raw rankings misattribute queue difficulty to agent skill.

**Tab 2 — Per-agent (adjusted).** OLS regression controlling for ticket category, channel, priority, product SKU, month, and assigned routing team. Residuals shrunk with Empirical Bayes (James–Stein). 95% confidence intervals shown. Only 3 agents sit with CIs entirely below the mean (and permutation tests show chance produces up to 3 flags at the 95th percentile).

**Tab 3 — Defect lot alerts.** A 3-sigma cumulative control chart monitoring replacement rate per (SKU, production month). Automatically flags Pulse 2 lots 2510, 2511, and 2512.

**Tab 4 — Data quality.** Quantifies timestamp correction for legacy helpdesk records (shifting resolved_at by +5.5h to eliminate 68% negative handle times), fallback lot join accuracy (94.9%), and SLA breach credits.

**Business outcome:**

> Detect bad production lots within ~27–34 days of first customer sale and immediately quarantine unsold inventory.
> On this dataset, the tool identifies **Rs 14.5 lakh** (Rs 1,451,124) in total excess replacement costs, of which **Rs 10.9 lakh** (Rs 1,087,175) was avoidable through timely lot-level holds. (A simpler ticket-level uniform baseline yields Rs 13.3 lakh excess and Rs 10.7 lakh avoidable, defining an expected range of **Rs 13.0 to 14.5 lakh**).

The replacement rate on bad lots is 41.7% to 47.0% (average 45.3%) vs ~7.6% on healthy lots. Average CSAT on bad lots falls to 2.7 vs 3.5 on healthy lots. The CSAT drop Priya Raman observed is driven by three defective production batches, not agent underperformance.

---

## What does one run cost, and what would a month cost at Vireo's volume (~650 tickets/week)?

**One run: Rs 0.**  
**One month: Rs 0.**

The pipeline is implemented entirely in open-source Python (`pandas`, `numpy`, `statsmodels`, `streamlit`). It makes **zero external API calls**, requires no paid LLM inference, and runs on standard local hardware.

The regex keyword rule (`sig_charge`) executes in milliseconds. The full OLS regression and lot control chart run in ~1.5 seconds. At full volume (~650 tickets/week or ~2,800 tickets/month), execution time remains under 5 seconds with zero marginal financial cost.

*Arjun's guidance was explicit: no per-ticket model calls at Rs 5 a pop. This solution costs Rs 0.*

---

## How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

### 1. Lot alert system
- **Coverage:** Verified on 4,380 of 4,402 Pulse 2 tickets (99.2% lot assignment coverage).
- **Join validation:** 94.9% accuracy on the fallback `customer_id + product_sku` merge_asof join (checked on 7,643 tickets where true `order_id` was known).
- **Detection timeline:** Alerts fired on **19 Nov 2025** (lot 2510, day 27), **29 Dec 2025** (lot 2511, day 34), and **25 Jan 2026** (lot 2512, day 32).
- **False-alarm rate:** Evaluated across 108 SKU-and-lot groups with at least 30 tickets. Exactly 3 groups were flagged—all three being the defective Pulse 2 batches (lots 2510, 2511, 2512). The other 105 lot groups produced **0 false alarms (0.0%)**. As a secondary check, weekly p-chart monitoring across 109 SKU-weeks yielded 1 alert on VA-SW-NX2 (0.9%), consistent with a low false-alarm rate, but based on a single event.

### 2. Agent adjustment (OLS regression & permutation test)
- **Model fit:** $R^2 = 0.106$ (`m.rsquared = 0.1063`). The model explains 10.6% of CSAT variance; 89.4% is residual noise, confirming that raw ticket CSAT is dominated by factors outside agent control.
- **Split-half reliability:** Evaluated over **300 random splits** (`n_iter=300`). Raw CSAT ranking has a reliability of 0.69 (driven by persistent queue sorting), whereas adjusted CSAT reliability drops to 0.25 because queue variance has been stripped away.
- **Permutation test (500 runs):** Under the null hypothesis of identical agent skill, randomly shuffling agent assignments flags an average of **0.95** agents and up to 3 agents at the 95th percentile. Exactly 3 agents were flagged in reality (A3004, A3005, A3006, all Chat Frontline). They are suggestive candidates for an informal manager conversation, not statistically confirmed underperformers.

### 3. Keyword rule (`sig_charge`) & Stratified evaluation (`validate.py`)
- **Evaluation sample:** Tested on a 120-ticket stratified sample (40 bad-lot Pulse 2, 40 good-lot Pulse 2, 40 non-Pulse-2) via `validate.py` using rule-assisted proxy labels.
- **Performance vs proxy labels:** 100% precision on the sample (expected by construction, as both rules require charging terminology) and **34.4% recall** (~34%, detecting 11 of 32 charge faults). Across all 4,402 Pulse 2 tickets vs agent replacement outcomes, precision is 67.7% and recall is 40.5%.
- **What it misses:** Customers reporting symptoms without the specific co-occurrence of "charg*" and "left/right" (e.g., "case is dead", "rapid battery drain", "right earbud won't turn on"). It serves as a rapid automated routing indicator, not an exhaustive hardware diagnostic. Independent manual hand-labelling remains an open item.

---

## Did you change, narrow, or push back on the client's ask?

**Yes, in three specific ways:**

1. **Pushed back on the "bottom ten" retraining mandate:** Priya asked for the bottom 10 agents to send to retraining. The data **indicates** that 7 of the raw bottom 10 are there largely because of routing (handling hardware complaints or courier waiting times). Only 3 agents are statistically below average, and permutation testing indicates that up to 3 agents can appear below average purely by chance. We recommended holding the majority of the training budget in reserve and conducting informal manager conversations instead.
2. **Corrected replacement unit economics:** Arjun's email suggested an arbitrary "Rs 2,500 all-in" cost per replacement. Applying Rs 2,500 to the roughly 800 excess replacements gives about Rs 20 lakh. Company policy specifies `unit cost + Rs 340 logistics`. For Pulse 2 (`unit_cost = Rs 1,480`), this equals **Rs 1,820**. We used Rs 1,820 throughout, yielding a defensible headline excess cost of Rs 14.5 lakh.
3. **Proactively introduced lot-level defect detection:** The original brief focused solely on agent performance and handle time. We discovered that the CSAT collapse was caused by three defective hardware lots (2510–2512). Identifying and quantifying this root cause provides an actionable operational lever saving ~Rs 10.9 lakh.

---

## What is wrong with what you are handing us?

1. **Sample volume limitation:** The export contains ~150 tickets/week against Vireo's reported ~650 tickets/week. All rupee figures represent an observed sample floor rather than an extrapolated total.
2. **Rule-assisted proxy labels in validation suite:** The 120-ticket sample in `validate.py` uses rule-assisted proxy ground truth rather than an independent manual human audit. Precision is high by construction, and true independent hand-labelling remains an open follow-up.
3. **Ticket-level vs unit-level replacement counts:** A customer submitting multiple tickets for the same defective unit could result in multiple replacement counts. Lot 2510 has 1,009 tickets vs 834 units sold; the ticket-level excess is a close proxy but not an inventory audit.
4. **Low adjusted ranking reliability ($r = 0.25$ over 300 iterations):** Removing queue effects exposes the high intrinsic variance of single CSAT ratings. The ranking is directional, not an exact scorecard.
5. **SLA credit overlap:** The pipeline calculates 1,064 SLA breaches totaling Rs 3.72 lakh in policy credits. Some breached tickets also received product refunds; potential overlap between customer credits and refunds has not been deduplicated.

---

## What did you deliberately leave out, and why?

| Left out | Why |
|---|---|
| Customer segmentation by `care_plus` | 30.2% of customers have Care+, but CSAT differences were trivial (3.31 vs 3.36); adding it adds complexity without changing operational conclusions. |
| Deep-dive into refunds | Total refunds equal **Rs 53.5 lakh** across 1,869 refunds (average refund **Rs 2,863**). However, 84% of ticket records lack refund data; replacement hardware cost (Rs 14.5L) is far cleaner and directly tied to the manufacturing fault. |
| Social & Voice SLA breakdown | While Email has the highest breach rate (12.2% vs 7.9% Chat), channel re-routing is a secondary operational lever compared to lot quarantine. |
| Complex NLP / LLM classifiers | High compute overhead and recurring cost for minimal gain; rule-based regex and OLS regression run instantaneously at Rs 0 cost. |
| Non-earbud batch analyses | Only Pulse 2 showed a sharp manufacturing defect wave; other SKUs remained stable. |

---

## Anything you built or found that nobody asked for?

1. **Automated lot-level cumulative control chart:** Detects defective batches within 27–34 days, unlocking Rs 10.9 lakh in avoidable costs (0 false alarms across 105 other lot groups).
2. **Empirical Bayes (James–Stein) shrinkage:** Stabilizes noisy agent residuals by pulling estimates toward the team mean based on sample size, preventing overreaction to small sample outliers.
3. **Cross-team handle time diagnosis:** Identified that Logistics (~1,465 min) and Returns (~1,465 min) median handle times reflect third-party courier waiting periods, not agent inefficiency (frontline teams average 21–24 min).
4. **Independent statistical validation suite (`validate.py`):** Encapsulates the 120-ticket stratified evaluation, permutation test, and 300-iteration split-half reliability in a single reproducible script.

---

## What did you use AI for?

**Role of AI assistant:**  
Used as an interactive pair-programmer to draft data transformations (`merge_asof` fallback logic, timestamp timezone correction), structure the statsmodels OLS formula and Empirical Bayes shrinkage equations, implement the Streamlit dashboard layout, and review analytical edge cases.

**What it helped with:**  
- Rapidly catching the 5.5-hour UTC timestamp offset in `legacy_fd` records that produced negative handle times.
- Vectorizing cumulative control chart bounds across SKU-lot combinations.
- Formulating the OLS specification and James-Stein shrinkage equations cleanly.

**What was discarded / tried and rejected:**  
1. **The Junk-IVR transcript detector:** Attempted a vocabulary-ratio filter on voice tickets to detect garbled speech-to-text outputs. Discarded after discovering it flagged 329 completely legitimate customer voice messages that happened to be brief or lacked arbitrary standard keywords. Reverted and kept all voice records intact.
2. **Initial weekly-only p-chart:** The weekly rule worked (first single-week alert on 1 Dec 2025), but we moved to cumulative lot-level monitoring ($n \ge 30$) because the core business question is which specific production lot is defective and when to hold warehouse stock.
3. **The `g` variable collision in exploratory analysis:** Cell 3 used `g` for lots, Cell 4 overwrote it with the permutation table, so the Cell 5 reprint of section C showed the wrong table. Caught and cleanly scoped without re-running the entire notebook.

**Paid API cost:** **Rs 0.** No paid APIs or remote inference used.

**Screen recording:** 3-minute video walkthrough covering the dashboard tabs, alert logic, and trade-offs.

---

## Your Public Google Drive Link

[Insert Google Drive link to repository zip, memo_to_priya.md, submission_form.md, and video recording]

---

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **The primary issue is hardware batch quality, not agent competence:** Pulse 2 lots 2510, 2511, and 2512 suffered a 41–47% failure rate. Chat Frontline agents received the brunt of these tickets, artificially dragging down their raw CSAT.
2. **Execution instructions:** Run `python -m vireo.pipeline` to verify core metrics, `python validate.py` to run the validation test suite, and `streamlit run app.py` to launch the interactive UI.
3. **Agent rankings are statistical estimates with low reliability ($r = 0.25$):** Only 3 agents have confidence intervals below average, and chance alone flags up to 3 agents (95th percentile). Do not penalize agents based on these rankings; conduct supportive manager check-ins and wait for post-defect data.

---

## Honest hours spent

**~4.5 hours** (within the 5-hour cap).

Breakdown:
- 1.5 h — Data auditing, timezone reconciliation, and exploratory data analysis
- 1.0 h — Regression modeling, Empirical Bayes shrinkage, and permutation testing
- 0.5 h — Lot control chart logic, avoidable cost calculations, and false-alarm testing
- 1.0 h — Pipeline modularization (`vireo/pipeline.py`), validation suite (`validate.py`), and Streamlit UI
- 0.5 h — Executive memo, submission documentation, and verification

---

## GitHub Repo Link

https://github.com/Mohithmcu/vireo-support
