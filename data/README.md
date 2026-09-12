# Data

The project uses the **AITEX Fabric Image Database (AFID)** introduced in:

Silvestre-Blanes et al., *A Public Fabric Database for Defect Detection Methods
and Results*, AUTEX Research Journal 19(4), 2019.
[DOI: 10.2478/aut-2019-0035](https://doi.org/10.2478/aut-2019-0035)

The authoritative downloads are hosted by AITEX:

- `https://www.aitex.es/wp-content/uploads/2019/07/Defect_images.7z`
- `https://www.aitex.es/wp-content/uploads/2019/07/NODefect_images.7z`
- `https://www.aitex.es/wp-content/uploads/2019/07/Mask_images.7z`

Run the preparation pipeline from the repository root:

```powershell
python scripts/prepare_dataset.py --download
```

This downloads and extracts the archives under `data/raw`, audits the source
files, creates deterministic train/validation/test assignments at the original
image level, and only then creates a patch manifest. Raw data, processed
manifests, and audit outputs are ignored by Git.

The source dataset is intended for research. Review the terms on the
[AITEX AFID page](https://www.aitex.es/afid/) before redistribution or commercial
use. The full dataset must not be committed to this repository.

