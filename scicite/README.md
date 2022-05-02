This archive contains the datasets for classifying citation intents in academic
papers (SciCite)

For details on the data refer to the NAACL 2019 paper:
["Structural Scaffolds for Citation Intent Classification in Scientific Publications"](https://arxiv.org/pdf/1904.01608.pdf).

Originally downloaded from [here](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz).

The data within this archive is formatted by us in the TSV format (each line
has 4 fields separated by tabs). The main citation intent label for each entry
is specified in the last column (label) while the citation context is specified
in the third column (string).
Example entry:

```
1e07b16fb589780721887f60271a8b1c2868fb8b>40c5741dc18ca03e625306b5bbf6622d097176a3_12
explicit        In addition to the three study species, yellow baboons (Papio
cynocephalus) and Sanje mangabeys (Cercocebus sanjei) are also present but,
being predominantly terrestrial, are rarely sighted from line transects (Rovero
et al. 2006, 2012), and the former occur only along the lowland forest edge.    background
```

The data is originally in the Jsonlines format (each line is a json object).
Note that the original data contains more information than what we provided
you in the TSV format. Make use of the extra information once you are
ready with the main implementation.

The main citation intent label for each Json object is specified with the label
key while the citation context is specified in with a string key. Example
entry:

```
{
 'string': 'In chacma baboons, male-infant relationships can be linked to both
    formation of friendships and paternity success [30,31].'
 'sectionName': 'Introduction',
 'label': 'background',
 'citingPaperId': '7a6b2d4b405439',
 'citedPaperId': '9d1abadc55b5e0',
 ...
}
```

You may obtain the full information about the paper using the provided paper
ids with the [Semantic Scholar API](https://api.semanticscholar.org/).

