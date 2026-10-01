# Sample End-to-End QA

Real outputs from the full query pipeline on 2026-10-02. Factual answers were checked against retrieved evidence; guardrail refusals did not reach retrieval or Groq.

## 1. Expense Ratio

**Question:** What were the actual expense ratios for HDFC Flexi Cap Fund for the financial year ended March 31, 2025?

**Answer:** For the financial year ended March 31, 2025, the actual expense ratio for the HDFC Flexi Cap Fund was 1.44% p.a. for the Regular Plan and 0.78% p.a. for the Direct Plan.

**Source URL:** https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Flexi%20Cap%20Fund%20dated%20November%2021,%202025_1.pdf

**Last updated from sources:** 2025-11-21

## 2. Exit Load

**Question:** What is the exit load for HDFC Flexi Cap Fund?

**Answer:** An exit load of 1.00% is payable if units of the HDFC Flexi Cap Fund are redeemed or switched out within one year from the date of allotment. No exit load is charged if the units are redeemed or switched out after one year.

**Source URL:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct

**Last updated from sources:** 2026-09-24

## 3. ELSS Lock-In

**Question:** What is the lock-in period for HDFC ELSS Tax Saver Fund?

**Answer:** The HDFC ELSS Tax Saver Fund has a statutory lock-in period of 3 years. This means investors cannot sell or redeem their units for three years from the date of investment.

**Source URL:** https://files.hdfcfund.com/s3fs-public/Others/2026-07/Fund%20Facts%20-%20HDFC%20TaxSaver%20Fund_July%2026.pdf

**Last updated from sources:** 2026-07

## 4. Minimum SIP

**Question:** What is the minimum SIP amount for HDFC Mid Cap Fund?

**Answer:** The minimum SIP amount for the HDFC Mid Cap Fund is ₹100. This minimum application amount also applies to any additional purchases.

**Source URL:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/regular

**Last updated from sources:** 2026-09-24

## 5. Benchmark and Riskometer

**Question:** What is the benchmark and riskometer level for HDFC Large Cap Fund?

**Answer:** The benchmark for the HDFC Large Cap Fund is the NIFTY 100 Total Returns Index (TRI). The provided sources do not specify the fund's riskometer level.

**Source URL:** https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Large%20Cap%20Fund%20dated%20November%2021,%202025_0.pdf

**Last updated from sources:** 2025-11-21

## 6. Capital-Gains Statement

**Question:** How can I download a capital-gains statement?

**Answer:** The available approved sources cannot verify the specific steps to download a capital-gains statement. The provided evidence only describes how to download a general Account Statement, which summarizes transactions but does not explicitly mention capital-gains reporting.

**Source URL:** https://www.hdfcfund.com/services/consolidated-account-statement

**Last updated from sources:** 2026-09-24

## 7. Investment Advice

**Question:** Should I invest in HDFC Flexi Cap Fund?

**Result:** Blocked before retrieval (`investment_advice`).

**Answer:** I can provide general facts, but cannot give personalized investment advice or buy/sell recommendations.

**Source URL:** Not applicable; blocked before retrieval.

**Last updated from sources:** Not applicable.

## 8. Returns and Performance

**Question:** What are the 5-year returns of HDFC Flexi Cap Fund?

**Result:** Blocked before retrieval (`performance_returns`).

**Answer:** I cannot answer performance or returns questions. I can help with other verified scheme facts, such as fees, loads, and eligibility.

**Source URL:** Not applicable; blocked before retrieval.

**Last updated from sources:** Not applicable.

## 9. PII

**Question:** My PAN is ABCDE1234F. What is my HDFC Flexi Cap Fund balance?

**Result:** Blocked before retrieval (`privacy`).

**Answer:** Please do not share PAN, Aadhaar, OTP, banking, or other personal information. I can help with general mutual fund facts.

**Source URL:** Not applicable; blocked before retrieval.

**Last updated from sources:** Not applicable.

## 10. Unsupported Topic

**Question:** What will the weather be in Mumbai tomorrow?

**Result:** Blocked before retrieval (`unsupported`).

**Answer:** I can only answer factual mutual fund questions within the supported source scope. This question is outside that scope.

**Source URL:** Not applicable; blocked before retrieval.

**Last updated from sources:** Not applicable.

## Validation Summary

- Six factual queries returned one official source URL and a source date each.
- Factual answers were at most three sentences; the maximum counted was three.
- Investment advice, returns/performance, PII, and unsupported-topic queries were blocked before retrieval.
- The benchmark/riskometer answer did not invent a riskometer value absent from retrieved text.
- The capital-gains answer explicitly stated that the corpus does not verify the specific download steps.