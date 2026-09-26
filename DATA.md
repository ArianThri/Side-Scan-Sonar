# Dataset availability

This repository does not redistribute SCTD imagery or Pascal VOC annotations.
The dissertation experiments used the acquired copy containing 357 images,
357 XML annotation files and 363 annotated objects. The published paper's
larger dataset count is contextual and was not reconstructed or imputed.

Obtain SCTD through an authorised source and place the acquired files under:

```text
data/raw/SCTD/
```

The dataset audit expects image/XML basename pairs. Run the audit before any
split or conversion command:

```powershell
python src\dataset_audit.py --dataset-root data\raw\SCTD --output-dir data\audit\current
```

The identifiers, hashes and summaries from the executed copy are retained in
`evidence/dataset_audit/` and `evidence/splits/`. They allow the acquired copy
and accepted split to be checked without publishing the image content.

Do not assume that another SCTD download has the same files solely because it
uses the same dataset name. Compare the audit manifest and object counts before
attempting reproduction.

