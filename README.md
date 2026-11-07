# natural-compounds
Does the Pretraining Corpus Matter for Molecular Property Prediction?

A model that predicts molecular properties can be trained on measured data alone, or first on a large collection of molecules that carry no measurements. We want to know whether the choice of that collection matters.
Labelled datasets in drug discovery are small because measuring a property is slow and expensive.
Pretraining on unlabelled data is one answer and it transformed language and protein models. For molecules the gains have been smaller and less consistent, and much of the work since has gone into new architectures and objectives. We think the data deserves a closer look. If the corpus is part of the limit then the choice of molecules is worth testing. 
That matters to anyone training property predictors on small assay data.
