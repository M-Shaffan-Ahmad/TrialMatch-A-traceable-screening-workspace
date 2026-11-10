# Build the IEEE report

From the repository root, run Tectonic with the IEEEtran class available:

```bash
mkdir -p output/pdf
tectonic --outdir output/pdf report/main.tex
```

Alternatively, upload the contents of the report directory to an IEEEtran LaTeX project
and build main.tex. The figures and validation table file are included. The paper labels
the measured validation results separately from the pending held-out comparison.
