# MIBiG biosynthetic classes: a candidate, documented, not adopted

25 November 2026.

The task would be: given a molecule, predict the broad biosynthetic class of
the gene cluster that makes it. PKS, NRPS, terpene, ribosomal, saccharide,
other.

**This is not the Pfam family task from the NaFM file.** That one is kept
separate, in section 6, and is not part of this package.

Nothing here has been trained on. This dossier exists so the data can be read
and judged, not so a decision can be handed over pre-made. Where I give an
opinion it is labelled as one.

## 1. Getting the data

```
uv run python scripts/prepare_mibig_inspection.py --fetch
```

Downloads the release, verifies its checksum, and writes four files under
`data/processed/mibig_inspection/`. `--check` does everything except write.
Both honour `NC_DATA_DIR`, and the script prints the directories it resolved.

| | |
|---|---|
| URL | `https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz` |
| sha256 | `986f1a69fb5eeb84fd7c0175ae921304c22fd5f8ed04c96d8c1c0c95a145837b` |
| size | 954,258 bytes |
| entries | 3,013 |

Fetched and verified from an empty data directory on 25 November 2026. The URL
answered, the bytes matched, the extraction reproduced every count below. A
fresh download compared byte for byte against our copy is identical, and the
server reports Last-Modified 15 November 2024, so the release has been stable
for a year.

The file name carries the release, `4.0`, and is served from a stable path.
That is the closest thing to a version pin this endpoint offers, so the
checksum is the real pin. A re-release under the same name would be caught by
it rather than silently absorbed. The download page itself is a single-page
app and serves no plain link, which is why the script hard-codes the file URL.

On terms of use: MIBiG's own Zenodo deposits are CC BY 4.0. This package ships
the script and not the data, so nothing is redistributed either way, and the
citation is the MIBiG 4.0 paper in Nucleic Acids Research.

## 2. What the four files are

| file | one row per | rows |
|---|---|---|
| `mibig_bgc.csv` | gene cluster entry | 3,013 |
| `mibig_compound.csv` | compound record under a cluster | 5,443 |
| `mibig_structure_bgc_class.csv` | (structure, cluster, class) triple | 5,485 |
| `mibig_inspection.json` | the counts below, regenerated | |

**There is no label table and that is deliberate.** A class in MIBiG belongs to
the gene cluster, not to the molecule. "The class of a molecule" has to be
constructed, and the obvious construction is to give each product all of its
cluster's classes and take the union across clusters. That union is a choice
with consequences, so it is not made here. The long table keeps the triples and
a disagreement between two clusters survives as two rows.

The long table's columns are `canonical`, `bgc`, `status`, `klass`,
`subclass`, `legacy_conversion`, `details`, `compound_name`. The last four
exist because dropping them hides what section 3 is about.

Canonicalisation is `natural_compounds.chem.canonicalize`, which keeps
stereochemistry. `mibig_compound.csv` also carries a stereochemistry-free form,
because the origin dataset carries no stereochemistry and the two cannot
otherwise be compared. Grouping is on the exact canonical string, with no
tautomer normalisation, so two entries writing the same molecule differently
stay separate.

## 3. The counts

Entries: 2,437 active, 377 retired, 199 pending.

Keeping active entries with a class and a parseable structure gives **3,118
distinct canonical structures**.

| class | distinct structures | triples | class rows on active entries | of those, converted from v3 |
|---|---:|---:|---:|---:|
| PKS | 1,556 | 1,711 | 1,010 | 0 |
| NRPS | 1,284 | 1,380 | 836 | 0 |
| other | 529 | 591 | 382 | **339 (88.7%)** |
| ribosomal | 183 | 194 | 326 | 0 |
| terpene | 276 | 298 | 169 | 0 |
| saccharide | 155 | 184 | 193 | 0 |

Three denominators because they are three different things. Structures carrying
the class under any cluster, rows of the long table, and class rows on the
entries themselves.

**The last column is the most useful thing in this document.** A class row
carries a `details` field saying how it got there, and a row reading "converted
from 'Alkaloid'" or "converted from v3 without extra details" came through the
v3-to-v4 converter rather than being annotated against the 4.0 vocabulary.

`other` is 88.7 percent migration. The remaining 43 rows are real annotations
with a substantive subclass: aminocoumarin 11, non-nrp beta-lactam 6,
nucleoside 6, phenazine 5, cyclitol 3, shikimate-derived 3, and a tail of ones
and twos. **Every other class has zero converted rows.** PKS, NRPS, ribosomal,
terpene and saccharide are clean 4.0 annotations throughout.

So `other` is not a biosynthetic class, it is where v3 entries landed when
their old label had no 4.0 equivalent, Alkaloid above all. Training on it would
be training a model to recognise which entries had not been re-annotated. The
long table carries `legacy_conversion` and `details` per row so this is visible
rather than buried.

Hybrid clusters: 456 of 2,437 active entries carry more than one class, 433
with two and 23 with three. NRPS + PKS alone is 336 of them.

Attribution: 1,944 clusters contribute at least one structure, 570 contribute
more than one, and the largest contributes 21.

## 4. The ambiguity, counted properly

I previously ran these numbers together and made the data look more
contradictory than it is. The breakdown matters.

| | count |
|---|---:|
| structures carrying more than one class | 833 of 3,118 |
| of those, because one cluster is itself hybrid | **825** |
| of those, because only of grouping across clusters | **8** |
| structures sitting under one cluster only | 2,903 |
| structures sitting under more than one cluster | 215 |
| of those, agreeing on their class set | 178 |
| of those, with genuinely **different** class sets per cluster | **37** |

The first split is the one I got wrong before. 825 of the 833 multi-class
structures are multi-class because a single cluster annotates several classes,
which is a hybrid cluster and is exactly what the annotation is for. Only 8
become multi-class purely through grouping across clusters. The multi-label
structure of this dataset is overwhelmingly biology, not bookkeeping.

A structure under two clusters that both say {NRPS, PKS} is not a disagreement
either, and 178 of the 215 agree like that. Only the 37 are cases where the
answer depends on which cluster you look at. Those 37 split further:

| kind | count |
|---|---:|
| nested, one cluster's set contains the other's | 27 |
| disjoint, no class in common | 8 |
| overlapping, neither contained | 1 |
| mixed, both relations present across three or more clusters | 1 |

The 27 nested cases are one entry annotated less completely than another, a
curation gap and not a contradiction. Each has exactly two distinct class sets.
One example: the same molecule is {PKS} under BGC0000043 and {NRPS, PKS} under
BGC0000976. The class the smaller set omits is NRPS 7 times, PKS 6, terpene 5,
saccharide 4, other 2, and two classes at once 3 times.

So the genuinely conflicting set is **8 structures out of 3,118**, and they are
worth naming because six of the eight dissolve once section 3 is applied.

| compound | the disagreement |
|---|---|
| aspergillic acid | BGC0001516 {NRPS} against BGC0002602 {other} |
| citrinin | BGC0000894 {other} against BGC0001338 {PKS} |
| coformycin | BGC0002039, BGC0002040, BGC0002435 {other} against BGC0002462 {saccharide} |
| cosmomycin B | BGC0000214 {PKS} against BGC0000650 {terpene} |
| ergotamine | BGC0001241 {terpene} against BGC0002232 {other} |
| leupeptin Ac | BGC0002095 {NRPS} against BGC0002134 {other} |
| nanangelenin A | BGC0002199 {other} against BGC0002731 {PKS} |
| schizokinen | BGC0002633 {other} against BGC0002683 {NRPS} |

Six of the eight pit a real 4.0 class against an `other` row, and every one of
those `other` rows is a v3 conversion. Ergotamine's BGC0002232 reads class
`other`, subclass `other`, details "converted from 'Alkaloid'". These are not
curators disagreeing, they are un-re-annotated v3 entries sitting beside
siblings that were re-annotated. 14 of the 37 divergences involve such a row.

Of the two that remain, cosmomycin B is a plain data error: BGC0000650's sole
compound is named "carotenoid" but carries formula C40H53NO14 and a structure
that canonicalises to the cosmomycin B of BGC0000214. The structure-level
grouping is what exposed it.

That leaves roughly one or two real biological disagreements in 3,118
structures. My earlier framing of this data as broadly ambiguous was wrong.

The number of clusters per divergent structure: two in 26 cases, three in 6,
four in 3, five in 1, seven in 1.

## 5. What the evidence fields do and do not say

Of the 3,449 active classed compound records, 866 carry a structured
structure-evidence annotation and 2,583 carry none. Where present the methods
are NMR 637, mass spectrometry 332, MS/MS 89, chemical derivatisation 39,
authentic standard 27, total synthesis 25, X-ray 18.

**An empty evidence list is not an absence of evidence.** It is an absence of a
structured annotation in this file. Every one of the 2,437 active entries
carries at least one literature reference, kept in `mibig_bgc.csv`. The
evidence the structure rests on may be perfectly good and simply not encoded.
My earlier framing of "three quarters carry no evidence" conflated the two and
was unfair to the data.

**The `quality` field, and what it does and does not mean here.** It reads
"questionable" on all 2,437 active entries, while all 303 graded entries, 233
high, 42 low and 28 medium, are retired or pending. I guessed earlier that this
was a schema default. It is not, and the real explanation is better evidenced.

The official v3-to-v4 converter hard-codes it. In `mibig-secmet/python-mibig`,
`convert_v3.py` line 191 sets `QualityLevel.QUESTIONABLE` unconditionally when
it builds each entry, with the inline comment "imported entries are
questionable until proven otherwise". No code path in that repository ever
assigns low, medium or high. A commit of 30 April 2024 relaxed validation at
that level, so an entry marked questionable may lack citations, evidence or
reviewers and still validate.

So for the active corpus the value is the converter's import floor, not a
per-entry assessment. Every one of the 2,437 carries `legacy_references`, a
field only the converter writes and which validation forbids above
questionable. An active entry marked questionable must not be read as curators
having judged it doubtful.

**But it is not merely an artefact either, and the flat negative is wrong.**
Quality is a required four-level enum that curators do set, and questionable is
its bottom rung. 42 questionable entries carry no `legacy_references`, 21 of
them pending, and 20 of those share the exact changelog signature of the 178
graded pending entries, one release at version 1 commented "MIBiG v4
annotathon". The 199 pending entries were graded in that annotathon and 21 came
out questionable, which for those is a judgement. One retired entry is
annotated "Retired, low quality entry". And BGC0001798, a v1.4-era legacy
entry, was re-curated and graded high, so an import origin does not pin an
entry at questionable forever.

The script copies the field through without interpreting it. Read it per
cohort, not globally.

`completeness` on active entries: unknown 1,859, complete 545, partial 33.

## 6. The Pfam variant, not part of this

Predicting Pfam domain families from structure, over MIBiG clusters joined to
their products. Different task, far more labels, far sparser, and the labels
come from deterministic HMM calls rather than curation.

The "96.3 percent" figure that has been attached to it is 1,408 of 1,462
clusters sharing at least one canonical structure between the NaFM-derived set
and MIBiG. It validates a partial structure-to-cluster join on that set. It
does not validate the Pfam vectors and it does not validate the biosynthetic
classes.

It also used a union of labels over a cluster's products, adopted by default
and never argued. Not put forward.

## 7. Where I stand, with the parts separated

My position has moved while writing this, and in the data's favour. I came in
expecting the labels to be murky. Most of what looked murky is the `other`
bucket and the v3 migration behind it, and once those are separated the five
real classes are clean.

**Facts, measured above.** Five of the six classes carry zero converted rows
and are 4.0 annotations throughout. `other` is 88.7 percent migration residue
and should not be treated as a class. Of the 833 multi-class structures, 825
are multi-class because one cluster is genuinely hybrid, which is biology, and
8 only through grouping. Genuine cross-cluster conflicts number one or two in
3,118 once the migration rows and one data error are set aside.

Against that: two classes are thin, ribosomal at 183 structures and saccharide
at 155, against PKS at 1,556. 570 clusters contribute several products each, so
products of one cluster must stay in the same fold or the label leaks.
Structures and scaffolds tie clusters together transitively, and taking
connected components under both ties leaves one component of 504 structures,
16.2 percent of the data, large enough that most protocols would have to pin it
to the training side.

**Hypothesis, not tested.** I expect a fingerprint baseline to do well here,
because the classes track gross structural features that a chemist reads off
directly. If that is right the room for a pretraining effect is small. I have
not measured it and I am not going to assert it. A peptide backbone does not
imply NRPS, strong performance at full supervision does not rule out a useful
difference in the low-label regime, and the AUROC thresholds I previously
proposed for deciding this had no justification behind them. I withdraw those
thresholds.

**Preference.** I no longer have a case for rejecting this, and I withdraw the
one I made. What is left is a judgement about where effort goes, and that is
Gabriel's.

If it is taken up, the shape the data argues for is five classes with `other`
dropped, multi-label rather than forced to one, grouping on connected
components of the cluster and scaffold ties, and the two thin classes reported
with their counts rather than averaged in. Nothing in this package imposes
that. It is what the numbers above suggest, and the long table keeps every
alternative open.

## 8. Open questions, unresolved

1. How the class should be attributed to a product. Union across clusters is
   the obvious rule and it is not obviously right. It matters for 8 structures
   out of 3,118, so it is a small question now, but it is still a choice and
   nothing in this package makes it.
2. Whether to keep the 43 real `other` rows as a class of their own, fold them
   into their subclasses, or drop them with the 339 converted ones. Dropping
   `other` entirely loses eleven aminocoumarins and six beta-lactams that were
   properly annotated.
3. Whether the two remaining genuine divergences are curation errors or real
   alternative pathways. That needs reading the entries against the literature,
   not counting them, and I have not done it.
4. How the thin classes should be handled. Ribosomal and saccharide have 183
   and 155 structures, which is enough to train on and thin enough that a
   per-fold metric on them will be noisy.
5. Whether a structure in MIBiG is also in the pretraining pools, and how much.
   Not measured. The connectivity column is there so it can be, and the origin
   dataset's experience says to measure it per draw and not per pool.
