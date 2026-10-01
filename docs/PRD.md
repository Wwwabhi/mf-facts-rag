# Product Requirements Document: Facts-Only Mutual Fund Assistant

## Goal
Build a small RAG-based FAQ assistant that helps users quickly find verified, factual information about selected HDFC mutual fund schemes from official public sources. The assistant should provide concise, citation-backed answers without offering investment advice or personalized financial guidance.

## Target users
- Retail investors looking for quick factual information about mutual fund schemes
- Users comparing factual scheme details without asking for recommendations
- Support and content teams handling repetitive mutual fund information queries

## In-scope features
- Answer factual questions using only approved official sources from HDFC AMC, SEBI, and AMFI
- Support a limited corpus covering four HDFC schemes:
  - HDFC Large Cap Fund
  - HDFC Flexi Cap Fund
  - HDFC ELSS Tax Saver Fund
  - HDFC Mid Cap Fund
- Retrieve relevant snippets from official scheme documents, factsheets, KIM, SID, FAQs, and investor-service guidance
- Provide short answers with one source citation and a visible “Last updated from sources: <date>” line
- Refuse advice, recommendation, and performance-prediction requests in a polite, facts-only manner
- Block PII and account-specific information requests safely
- Expose the RAG pipeline separately from the UI so it can be inspected and tested independently

## Out-of-scope features
- Investment recommendations or buy/sell decisions
- Portfolio construction or personalized financial advice
- Return calculations, performance comparisons, or future return predictions
- Ranking schemes or identifying the “best” fund
- Personalized account information or transaction execution
- User account management or portfolio tracking
- Collection of personal or sensitive information
- Use of unofficial, third-party, or blog-based sources as factual knowledge

## Example user questions
- What is the expense ratio of HDFC Large Cap Fund?
- What is the exit load of HDFC Flexi Cap Fund?
- What is the minimum SIP amount for HDFC Mid Cap Fund?
- What is the lock-in period for HDFC ELSS Tax Saver Fund?
- What is the benchmark of the scheme?
- What is the riskometer level of the scheme?
- What is the minimum investment amount?
- What is the investment objective of the scheme?
- How can I download a capital-gains statement?
- How can I download an account statement?

## Success criteria
- The assistant answers factual mutual fund questions using the selected official source corpus
- Answers are relevant, concise, and limited to 3 sentences or fewer
- Each factual answer includes exactly one correct official source link
- Every answer includes the required “Last updated from sources: <date>” line
- Advice and portfolio questions are refused appropriately
- PII and account-specific requests are safely blocked
- Unsupported questions return a clear “not found / cannot verify” response instead of fabricated facts
- No third-party source is used in the knowledge base
- The RAG pipeline and UI are independently testable

## Constraints
- Only official public sources from HDFC Mutual Fund / HDFC AMC, SEBI, and AMFI may be used
- The project is limited to a small corpus and a small set of selected schemes
- Answers must be based only on information available in the approved sources
- The assistant is not intended to answer questions outside the selected corpus or unsupported facts
- The system must remain facts-only, not investment advice

## Safety and privacy requirements
- Do not accept, request, process, or store PII or account-specific information
- Block requests containing PAN, Aadhaar, OTP, bank details, mutual fund account numbers, phone numbers, or email addresses
- If such information is provided, respond with a safe message explaining that only general factual scheme information can be provided
- Do not encourage users to share sensitive data through the assistant

## Answer requirements
- Use only information retrieved from approved official sources
- Keep responses brief and factual, ideally within 3 sentences
- Include exactly one clear source link, sourced from retrieved metadata
- Include the text: “Last updated from sources: <date>”
- If a fact is not supported by the available corpus, say the information could not be verified from the available sources
- Do not make unsupported claims, infer missing facts, or fabricate a citation
- For investment advice or performance questions, politely refuse and direct users to official investor education resources where appropriate

## Disclaimer
Facts-only. No investment advice. This assistant provides factual information from selected official public sources and does not provide investment, financial, or portfolio advice.
