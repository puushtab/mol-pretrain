# Project memory

Durable decisions, constraints, findings and open questions. Read this at the
start of a major task. Append rather than rewrite, and date what you add.

Detail lives in `docs/protocol.md` and the stage documents. This file is the
index of what is settled and what is not.

## Constraints

- Never commit, never push. Leonard commits himself. No attribution trailer.
- No local experiments. Training, evaluation and anything that produces a
  number we would report runs on MIT ORCD Engaging.
- The frozen split files are frozen. `make_splits.py` refuses to rewrite one
  without `--force`. A rewrite silently invalidates every earlier cell.
- The 200 cells under `runs/finetune/` are the campaign of 10 November and
  stay as they are. New work goes to a new output directory.
- Writing rules in CLAUDE.md apply to every markdown file and docstring.

## Settled findings

**The pretraining corpus is a CID-ordered prefix, not a random draw.**
943,834 of 944,296 lines of `cleaned_pubchem_1MM.smiles` map to CIDs 1 through
1,111,175, none above, eight descents all from duplicate SMILES. CheMeleon's
paper says "randomly selecting 1 million molecules from PubChem". Those CIDs
are the 2004 and 2005 deposits, which is why 73 percent of BBBP and 93 percent
of ESOL sit inside. Consequence: Stage C measures a size effect within that
corpus, not within PubChem. 16 November 2026.

**Pretrained initialisation helps, and how much depends on a family nobody
declared.** Raw, fourteen of thirty-two paired comparisons against scratch
clear 0.05. Holm over thirty-two leaves one, ESOL at fifty labels. Holm over
the eight released-against-scratch leaves three. Neither family was
pre-registered, so both are post hoc. 16 November 2026.

**The similarity analysis resolves nothing, and the floor is not why.** The
pre-registered primary cell is 0.0236 with an interval from -0.0503 to 0.0975.
The one positive corpus-similarity cell appears at nearly the same size in
chemprop_descriptors against chemprop, where neither arm ever saw the corpus.
Under Webb six-point weights the 0.0625 sign-flip floor drops to 0.00026, and
the answer does not change. Stronger still: that cell comes out at p 0.0237 in
the control contrast against 0.0442 in the treatment one, so the slope is
larger where no model saw the corpus. 16 November 2026.

**Descriptor augmentation stays competitive.** At full labels CheMeleon leads
it by 0.0083 AUROC on BBBP and 0.0064 RMSE on ESOL, signs mixed across splits.
Not evidence of equivalence, which would need a margin.

**Stage B was never replicated and one seed did four jobs.** Until 16 November
the split seed also drove the scratch initialisation, the head, the dataloader
shuffle and the label draw. Separated now, with the label draw kept on the
split seed.

**Splits buy what seeds do not.** At rho 0.5, three seeds at five splits
multiply the interval half-width by 0.816 while ten splits multiply it by
0.576. The sign-flip floor, the t quantile and the Satterthwaite degrees of
freedom are all functions of the number of independent units alone.

**The campaign is CPU-cheap.** Measured: five to seven minutes a cell for the
two 2048-wide arms on four cores. The enlarged campaign is about 384
core-hours, which is a few hours on the free tier. `mit_preemptable` gives
1024 cores and a 48-hour wall. GPU throughput is capped at two cards.

**The splitter is worth more than the effect.** ChemBERTa-3 reports a D-MPNN
on BACE at 0.878 under MoLFormer's scaffold split and 0.626 under DeepChem's,
and on BBBP 0.930 against 0.661. Our released-against-scratch BBBP gain is
0.0247. The pairing removes this from the internal comparison and not from
what the number means outside it. Relayed from a preprint we could not fetch,
DOI 10.26434/chemrxiv-2025-4glrl-v2, so verify before quoting. 16 November
2026.

**The family dispute was partly a category error.** Rubin (DOI
10.1007/s11229-021-03276-4) distinguishes disjunction testing, which needs
alpha control, from individual testing, which does not even when the tests sit
side by side. "Pretraining helps somewhere" is a disjunction claim over the
eight primary cells. "It helps on ESOL at fifty labels" is an individual claim
and needs no adjustment. His exchangeability condition also rules out pooling
BBBP and ESOL into one family, since Amendment 5 already says they behave
differently. Declare both claims, keep them apart. 16 November 2026.

**The BBBP full-label headline does not survive replication.** Ten splits by
three training seeds give +0.0088 with an interval from -0.0006 to +0.0182,
against the November campaign's +0.0247 from -0.0150 to +0.0345. ESOL holds at
every budget in both. See [stage-b2-results.md](stage-b2-results.md).
16 November 2026.

**The BBBP fall is the new splits, not the training seeds.** Withdrawn: the
earlier entry here attributed it to training noise via rho. On splits 0 to 4
with three seeds the effect survives at +0.0174 [+0.0097, +0.0252]. On splits
5 to 9 it is +0.0002. Welch between the groups p 0.037. Averaging three seeds
instead of one *raises* the all-ten estimate by +0.0085, so seeds cannot
produce a fall. 17 November 2026.

**Rho carries no usable precision and its design conclusion was backwards.**
Six of eight bootstrap intervals span more than half of [0,1], and at BBBP
full labels the interval is 0.52 to 1.00 with the clamp firing in 59.4 percent
of resamples. Splits dominate seeds at every rho, because the variance of the
mean is sigma_split^2/S + sigma_seed^2/(S K) and both terms carry S. Do not
cite the measurement as novel. 17 November 2026.

**The two campaigns do not reproduce each other.** Of 96 cells with genuinely
matching configuration, 19 agree to 1e-12 and the worst gap is 0.60 RMSE. The
laptop is ARM, Engaging is x86. Any November-to-cluster comparison carries a
platform difference that no decomposition inside the data can remove.
17 November 2026.

## Open questions

- Whether the ORCD account authenticates at all. The pilot of 10 November
  recorded `Permission denied (publickey,keyboard-interactive)`. Nothing
  downstream can start until `slurm/preflight.sh` has run.
- Whether the fourteen Stage C checkpoints exist. `docs/models.md` lists
  thirteen and names `pubchem_n100000_s0` as not exported. There is no
  `runs/pretrain/` here. Ask Gabriel.
- The 462 released corpus lines that match nothing in the PubChem extract.
- Whether `probe-results.md` 0.9317 and the fine-tuning table's 0.9302 are the
  same quantity.
- The value of rho, the seed's share of the spread. Three training seeds will
  measure it, and nobody has measured it in molecular property prediction.
- How far below ten the effective number of independent splits sits, given
  that test sets overlap across scaffold draws.
- Whether the acyclic exclusion changes any conclusion. Not answerable under
  the current split design.

**Stage B3 is the campaign the conclusions rest on.** 1,560 cells, three
tasks by ten splits by three training seeds, all under one frozen code version
`d361f708393708ac`, verified by twelve checks. Released against scratch, Holm
within the declared family: five of eight incumbent cells survive, and all
four lipophilicity cells. ESOL survives at every budget. BBBP survives only at
500 labels. The BBBP full-label cell is +0.009096 with a lower bound of
+0.000029, which is on the boundary and does not survive Holm at 0.148.
17 November 2026.

**Lipophilicity is the clearest result in the campaign.** All four budgets
significant raw and under Holm, consistent direction, and it is the benchmark
chosen by a written rule before any model touched it. It is also the one with
18.4 percent corpus overlap against BBBP's 72.7 and ESOL's 93.3.
17 November 2026.

**Le gros encodeur est un handicap sur les régressions, et il mange le gain de
pré-entraînement.** L'écart public released contre chemprop se décompose
exactement en (released − scratch) plus (scratch − chemprop). Le second terme
est négatif aux huit cellules de régression, significatif à cinq, Holm à deux.
Sur la lipophilicité à pleine supervision: +0,052 de pré-entraînement, −0,050
d'architecture, +0,0024 au total. Le modèle publié ne bat donc pas un chemprop
standard sur les régressions, et cela ne dit rien contre le pré-entraînement.
Décomposition reproduite par deux implémentations indépendantes.
17 novembre 2026.

**Dix scaffold splits en valent entre 4,5 et 6,7.** Les jeux de test se
recouvrent. Jaccard moyen par paire 0,061 sur BBBP, 0,138 sur ESOL, 0,056 sur
la lipophilicité, d'où 6,46, 4,47 et 6,67 splits effectifs et des intervalles à
élargir de 1,24, 1,50 et 1,22. Cela répond à la question ouverte plus haut.
Appliqué, ESOL à pleine supervision et BBBP à pleine supervision tombent. Le
facteur suppose une corrélation échangeable égale au recouvrement, c'est un
ordre de grandeur. 17 novembre 2026.

**La lipophilicité est le seul résultat qui passe tout.** Quatre budgets sur
quatre survivent à Holm dans sa famille déclarée, au retrait de n'importe quel
split, à l'élargissement par le design effect et au retrait des unités ayant
changé de machine. BBBP n'en garde qu'un sur quatre, le budget 500.
17 novembre 2026.

**La pente de similarité existe sur la lipophilicité et sa spécificité n'est
pas établie.** Positive aux quatre budgets pour released contre scratch, et
elle survit à l'entrée conjointe de la proximité au jeu d'entraînement, qui
était la principale explication concurrente. Mais sous la spécification
complète le contraste de contrôle, où aucun bras n'a vu le corpus, donne à
pleine supervision +0,114 contre +0,098 pour le traitement. L'excès combiné
vaut +0,207 avec un intervalle de −0,017 à +0,431. C'est le signal le plus fort
du projet et ce n'est pas un résultat. 17 novembre 2026.

**BBBP et ESOL ne peuvent pas répondre à la question de similarité.** 81,1 et
87,4 pour cent de leurs molécules ont un voisin identique dans le corpus, donc
le régresseur n'a pas de variation. Ce n'est pas un manque de puissance et
aucune taille d'échantillon ne le répare. La lipophilicité n'a que 20,5 pour
cent au plafond. 17 novembre 2026.

**Correction, le mélange de machines porte sur 14 unités et non 35.** Une
préemption sur mit_preemptable requeue en gardant le job id et change de nœud.
Trente-cinq unités sur 360 ont l'un de leurs quatre bras neuronaux sur une
autre machine, 14 ont released et scratch séparés. Seul BBBP à pleine
supervision le ressent, Welch p 0,030 entre unités mélangées et propres.
17 novembre 2026.

**Une random forest bat CheMeleon sur BBBP en petit régime.** −0,1451 AUROC à
50 labels et −0,0866 à 100. À 50 labels un tiers des cellules released et 57
pour cent des cellules chemprop finissent sous le hasard. La forest garde son
avantage de 12 pour cent de labels, qui n'explique pas un tel écart.
17 novembre 2026.

**Correction de l'entrée ci-dessus sur la similarité, après vérification
adversariale.** Moyenner les quatre budgets avant de tester était la mauvaise
agrégation. Elle mélange un régime où le contrôle égale le traitement, 50 et
100 labels, avec un régime où il vaut trois à six fois moins, 500 et pleine
supervision. Marche brute au plafond sur la lipophilicité, appariée sur le
split: traitement +0,086 et +0,065 contre contrôle +0,025 et +0,012, différence
p 0,121 et 0,068. Sur ESOL à 500 la différence tranche, p 0,002, sur une
quinzaine de molécules sous le plafond par split. Si la spécificité au corpus
existe, elle vit aux budgets élevés. Elle n'est pas établie sous l'appariement
par split. Un regroupement par molécule la rend significative et tient les dix
splits fixes, donc il répond à une question plus étroite. 17 novembre 2026.

**Le verdict de multiplicité dépend de la famille, et la famille déclarée
compte.** Famille lipophilicité de quatre, celle du document de consolidation:
quatre cellules sur quatre. Famille de douze, les trois tâches ensemble: deux.
Famille de vingt-quatre avec les contrastes contre chemprop: ESOL à pleine
supervision tombe aussi. Citer un décompte sans nommer sa famille ne veut rien
dire. 17 novembre 2026.

## Révision du 18 novembre 2026, après audit externe

**Le facteur d'élargissement n_eff = 10/(1+9J) est retiré.** Le recouvrement
des jeux de test est mesuré, la corrélation entre les estimateurs par split ne
l'est pas, et poser l'égalité des deux était une hypothèse ajoutée que rien ne
soutient. Les entrées du 17 novembre qui s'appuyaient dessus sont caduques. Le
remplacement est un bootstrap en grappes de molécules puis de splits, avec
recalcul de la vraie métrique à chaque tirage. Un premier essai de bootstrap
rééchantillonnait une erreur absolue moyenne contre un intervalle de RMSE, ce
qui était le mauvais estimand.

**ESOL est le résultat le mieux établi, pas la lipophilicité.** Quatre budgets
sur quatre passent cinq tests sur cinq: Student apparié, Holm dans la famille
déclarée, retrait d'un split, test des signes, bootstrap en grappes. La
lipophilicité en passe cinq sur cinq à 100 et 500, quatre sur cinq à 50 et à
pleine supervision. L'entrée du 17 novembre disait l'inverse, parce que
l'heuristique retirée pénalisait justement ESOL, dont les jeux de test se
recouvrent le plus. 18 novembre 2026.

**BBBP à 500 labels est un résultat.** Quatre tests sur cinq, dix splits
favorables sur dix, p des signes 0,002. Le seul échec est le bootstrap sur
AUROC, mesure mal fondée sur une statistique de rang puisque rééchantillonner
des molécules change la prévalence. Dire « nous n'avons aucun résultat sur
BBBP » était faux. À pleine supervision, en revanche, un test sur cinq.
18 novembre 2026.

**Le bénéfice existe hors des molécules déjà vues.** Lipophilicité restreinte à
Tanimoto strictement sous 1: +0,0514 [+0,0072, +0,0956] à 500 labels et +0,0251
[+0,0101, +0,0402] à pleine supervision. Le gain ne se concentre donc pas
exclusivement sur les correspondances exactes. 18 novembre 2026.

**Le changement de régime entre budgets n'est pas établi.** Excès aux budgets
500 et complet moins excès aux budgets 50 et 100: +0,058 [−0,031, +0,147] sur
la lipophilicité, +0,208 [−0,157, +0,572] sur ESOL. Et l'excès ajusté s'annule
sur la lipophilicité, +0,016 et −0,001, alors qu'il survit sur ESOL où le
support n'est que de douze molécules par split. Aucune tâche ne donne un excès
à la fois bien supporté et robuste à l'ajustement. 18 novembre 2026.

**Un Tanimoto de 1 ne vaut pas appartenance au corpus.** Sur la lipophilicité,
838 molécules au plafond pour 771 réellement dans le corpus, donc 67 faux
positifs et aucun faux négatif. Le plafond surestime l'appartenance d'environ
8 %. 18 novembre 2026.

**La forest n'a pas plus de labels au total.** Elle ajuste ses arbres sur train
plus validation, et les bras neuronaux utilisent aussi les labels de validation
pour choisir leur checkpoint. L'avantage porte sur les labels servant à
ajuster directement des paramètres, pas sur la quantité acquise. 18 novembre
2026.

**Correction du bootstrap, 18 novembre 2026.** Les deux entrées ci-dessus qui
s'appuyaient sur un bootstrap en grappes reposaient sur un script faux. Sa
population de molécules était le jeu de test du split 0 seulement, au lieu de
l'union des dix. Les neuf autres splits étaient donc évalués sur leur
intersection avec le split 0, soit 8 à 38 molécules sur 205. Et np.unique sur
le tirage avec remise jetait les multiplicités, donc ce n'était pas un
bootstrap. Deux splits passaient sous le seuil et disparaissaient, bbbp 8 et
esol 4. Corrigé: union des molécules testées, poids de multiplicité partagés
entre tous les splits et les deux bras, métrique recalculée par graine avant
agrégation. Plus un contrôle d'identité à poids unitaires qui retrouve
l'estimation originale à 1,1e-16.

Conséquence. Huit cellules sur douze passent les cinq tests de robustesse, pas
six. BBBP à 500 labels passe de quatre sur cinq à cinq sur cinq, et la
lipophilicité à pleine supervision aussi. **Il n'y a donc pas de classement à
faire entre ESOL et la lipophilicité**, les deux régressions tiennent. Les deux
classements que j'ai publiés étaient chacun l'artefact d'un défaut de méthode,
l'élargissement heuristique d'abord, ce bug ensuite.

Règle à retenir: une analyse de rééchantillonnage sans contrôle d'identité à
poids unitaires ne doit pas être publiée. Le contrôle aurait attrapé les deux
fautes immédiatement.

**Le 44 et le 67 sont tous les deux justes.** Molécules au plafond de Tanimoto
sans appartenance réelle au corpus: 67 sur les 4 200 du jeu lipophilicité, 44
sur les 2 721 effectivement évaluées. Citer l'un ou l'autre sans nommer la
population est ce qui crée le désaccord. 18 novembre 2026.
