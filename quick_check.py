from backend.app.modules.evidence_retrieval import retrieve_evidence
from backend.app.models.schemas import Claim

c = Claim(claim_id="c1", claim_text="The Eiffel Tower was built in 1887")
ev = retrieve_evidence(c)
print("Source:", ev.source)
print("Text:", ev.text[:200])
