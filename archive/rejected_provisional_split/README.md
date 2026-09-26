# Rejected provisional split

`create_splits.py` generated the earlier class-aware split that allowed 12
exact-duplicate groups to cross partitions. It is retained only to document the
corrective history. Accepted experiments use `src/create_group_aware_splits.py`
and the zero-leakage identifiers in `evidence/splits/group_aware_v1/`.

