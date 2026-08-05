### A. Pair-level summary

Over all 134,209,536 unordered pairs of the 16,384 bits (268,419,016 have I(i;j) > 0).

| quantity | value |
|---|---|
| pairs with `I(i;j)/min(H_i,H_j)` > 0.5 | 174,881 |
| pairs > 0.9 | 26,123 |
| pairs > 0.99 | 5,444 |
| two-way implication pairs (identical support) | 2,808 |
| strictly one-way pairs, P(i\|j)=1 and P(j\|i)<0.9 | 268,122 |
| median `I(i;j)/min(H_i,H_j)` over nonzero pairs | 0.0002 |

### B. Strongest two-way pairs (identical support = exact duplicates)

`P(i|j) = P(j|i) = 1`, so `I(i;j) = H_i = H_j` and the ratio is 1.000 under either normalization. Ranked by MI, i.e. by how much entropy is duplicated.

| frag i | r | frag j | r | mols | MI (bits) |
|---|---|---|---|---|---|
| `CCC=CCC=CCC=CCC` | 6 | `C=CCC=CCC=CCC=CCC` | 6 | 7370 | 0.1075 |
| `c1ccc2c(c1)OCO2` | 3 | `ccc1c(cc)OCO1` | 4 | 4515 | 0.0721 |
| `CCC=CCC=CCCC(=O)O` | 5 | `CC=CCC=CCC=CCCC(=O)O` | 6 | 3181 | 0.0539 |
| `ccccs` | 2 | `cccsc` | 2 | 2885 | 0.0496 |
| `Cc1ccoc1` | 3 | `cocc(c)C` | 2 | 2714 | 0.0472 |
| `CC=CCCCC(=O)OCC(COC)OC(C)=O` | 6 | `C=CCCCC(=O)OCC(COC(C)=O)OC(=O)CC` | 6 | 2477 | 0.0437 |
| `CCC=CCCCCC(=O)OC` | 5 | `CC=CCC=CCCCCC(=O)OC` | 6 | 2410 | 0.0427 |
| `CC=CCCCCC(=O)OCC` | 5 | `CCCCCC(=O)OCC` | 4 | 2409 | 0.0427 |
| `CCn1c(C)cccc1=O` | 3 | `cc(C)n(CC)c(c)=O` | 2 | 2390 | 0.0424 |
| `CC(C)c1cccc(=O)n1C` | 4 | `CC(C)c1ccccn1C` | 3 | 2346 | 0.0417 |
| `CC=CCCCCC(=O)OCC(CO)OC` | 6 | `CCCCCC(=O)OCC(CO)OC` | 5 | 2192 | 0.0394 |
| `CC=CCCCCCCC(=O)OCC` | 6 | `CCCCCCCC(=O)OCC` | 5 | 2118 | 0.0383 |

### C. Strongest strictly one-way implications

`P(i|j) = 1` (support(j) ⊆ support(i)) but `P(j|i) < 0.9`. Bit *j* is the rarer, more specific fragment; *i* still fires without it, so `H(i|j) > 0` and the slot is **not** free. Ranked by MI.

| implied bit i | r | mols i | implying bit j | r | mols j | P(j\|i) | MI (bits) | MI/H_i |
|---|---|---|---|---|---|---|---|---|
| `ccc` | 1 | 154851 | `ccccc` | 2 | 116393 | 0.752 | 0.5266 | 0.599 |
| `[C] atom` | 0 | 189958 | `CC` | 1 | 132532 | 0.698 | 0.4961 | 0.524 |
| `CCC` | 1 | 73629 | `CCCC` | 2 | 64041 | 0.870 | 0.4599 | 0.781 |
| `CCCCC` | 2 | 65943 | `CCCCCCC` | 3 | 56152 | 0.852 | 0.4175 | 0.760 |
| `cc(c)O` | 1 | 83605 | `cc(c)OC` | 2 | 63420 | 0.759 | 0.4072 | 0.639 |
| `CCC` | 1 | 73629 | `CCCCC` | 2 | 58361 | 0.793 | 0.4028 | 0.684 |
| `[C] atom` | 0 | 187401 | `CC` | 1 | 110482 | 0.590 | 0.3943 | 0.418 |
| `[C] atom` | 0 | 189958 | `CCC` | 1 | 109787 | 0.578 | 0.3849 | 0.406 |
| `CCCCC` | 2 | 58361 | `CCCCCC` | 3 | 48830 | 0.837 | 0.3778 | 0.745 |
| `CCCCCCC` | 3 | 56152 | `CCCCCCCCC` | 4 | 46962 | 0.836 | 0.3686 | 0.745 |
| `CC` | 1 | 121096 | `CCC` | 1 | 73629 | 0.608 | 0.3637 | 0.464 |
| `CCC` | 1 | 103128 | `CCCCC` | 2 | 65943 | 0.639 | 0.3619 | 0.503 |

### D. Largest exact-duplicate groups

| group size | mols | example fragments |
|---|---|---|
| 28 | 246 | `CC(C)OC(C)(C)O`, `CNC1C(O)CC(OC(C)C)(C(…`, `CC(=O)NC1C(O)CC(OC(C)…` |
| 19 | 302 | `CC(=O)CCCC=Cc1cc(O)cc…`, `C=Cc1cc(O)c(C)cc1C`, `CCC=Cc1cc(O)c(C)c(O)c…` |
| 18 | 278 | `CCC1C=CC(C)CC1O`, `CCCC1C=CC(CC)C(O)C1`, `CCC1C=CC(C)C(O)C1` |
| 13 | 406 | `cc1c(c)C(=O)C2(O1)C(O…`, `COc1cc(OC)c2c(c1Cl)OC…`, `C=C(O)C1(C(C)C)Oc2c(C…` |
| 13 | 314 | `CC(C)C1CCC2C(C=CN3CNC…`, `CC(C)C1CCC2C(C=CN(C)C…`, `cc(o)CC(C(O)CO)C1CCC2…` |
| 11 | 390 | `C=CN(CN)C(C)C`, `C=CN1CNCC1C(C)C`, `CCC1C=CN2CNCC2C1C` |
| 10 | 435 | `CCCCCCCOC(COC(C)=O)CO…`, `CCCCCCOC(COC(=O)CC)CO…`, `CCCOC(COC(C)=O)COC(C)…` |
| 9 | 473 | `cc(C)c1c(c)C(C)(C)OC1…`, `Cc1ccccc1C(C)(C)O`, `ccc(c(c)C)C(C)(C)O` |
