# ETHICS.md — ethics path (resolve before submission)

The proposal's Section 6 has an unresolved contradiction that a supervisor will
flag. Resolve it by choosing ONE path and ticking the matching box on the form.

## The contradiction
- The proposal says **no ethics approval needed** because data is synthetic.
- BUT it also mentions **optional anonymised survey collection**, which is
  human-subject primary data and **does** trigger ethics approval.
- Both boxes on the form are currently blank.

## Recommended path: CLEAN NO (default for this build)
Drop the optional survey component entirely. The pipeline as built uses **only**
synthetic data generated from published clinical ranges, plus a **public,
already-anonymised** real dataset for hold-out validation (Stage 07).

- Tick **"NO, Not Required"** only if your institution treats use of a public,
  fully-anonymised dataset as not-human-subjects. **Confirm with your supervisor**
  — many institutions still require the secondary-data ethics route even for
  public datasets.
- If in doubt, take the YES path below; it is cheap insurance.

## Alternative path: YES (if you keep surveys OR if public-data use counts)
File the EP4DIS Research Ethics Statement & Application. Required if you collect
any survey responses or if your institution counts secondary public-data use as
human-subject data.

## Action
- [ ] Decide path with supervisor (Kate / Julien).
- [ ] Tick the correct box on the proposal form.
- [ ] Remove the survey wording from the proposal if taking the CLEAN NO path,
      so the document is internally consistent.
