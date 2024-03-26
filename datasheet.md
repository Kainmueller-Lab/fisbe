---
layout: default
title: Datasheet
---


[[**`Paper`**](https://)] [[**`Project`**](./index)] [[**`Documentation`**](./datasheet)] [[**`Metrics`**](./index#metrics)] [[**`Leaderboard`**](./index#leaderboard)] [[**`BibTeX`**](./index#citation)] [[**`Changelog`**](./changelog)]

# Datasheet

On this page we answer the Datasheet for Datasets questionnaire [[1]](#references) to document the FlyLight Instance Segmentation Benchmark (FISBe) dataset. It contains information about motivation, composition, collection, preprocessing, usage, licensing as well as hosting and maintenance plan.

## Motivation

**For what purpose was the dataset created?** *Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.*

Segmenting individual neurons in multi-neuron light microscopy (LM) recordings is intricate due to the long, thin filamentous and widely branching morphology of individual neurons, the tight interweaving of multiple neurons, and LM-specific imaging characteristics like partial volume effects and uneven illumination.<br>
These challenging properties reflect a key current challenge for deep-learning models across domains, namely to efficiently capture long-range dependencies in the data. While methodological research on this topic is buzzing in the machine learning community, to date, respective methods are typically benchmarked on synthetic datasets.<br>
To fill this gap, we created the FlyLight Instance Segmentation Benchmark dataset, to the best of our knowledge, the first publicly available multi-neuron LM dataset with pixel-wise ground truth and the first real-world benchmark dataset for instance segmentation of long thin filamentous objects.<br>


**Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**

This dataset was created in a collaboration of the Max-Delbrueck-Center for Molecular Medicine in the Helmholtz Association (MDC) and the Howard Hughes Medical Institute Janelia Research Campus. More precisely, the Kainmueller lab at the MDC and the Project Technical Resources Team at Janelia.<br>


**Who funded the creation of the dataset?** *If there is an associated grant, please provide the name of the grantor and the grant name and number.*

Howard Hughes Medical Institute Janelia Research Campus and Max-Delbrueck-Center for Molecular Medicine in the Helmholtz Association (MDC) funded the creation of the dataset.<br>


## Composition

**What do the samples** (Note by authors: We changed **instances** to **samples** when refering to images of the dataset to not use the term ambiguously; instead we only use **instances** to refer to **object instances** in images) **that comprise the dataset represent (e.g., documents, photos, people, countries)?** *Are there multiple types of samples (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.*

The dataset consists of 3d multi-neuron multi color light microscopy images and their respective pixel-wise instance segmentation masks.<br>
The "raw" light microscopy data shows neurons of the fruit fly Drosophila Melanogaster acquired with a technique called Multi-Color FlpOut (MCFO) [[2,3]](#references).
Fruit fly brains of different transgenic lines (e.g. GAL4 lines [[4]](#references)) were imaged, each transgenic line tags a different set of neurons.
Multiple MCFO images of the same transgenic lines each express (show) a stochastic subset of the tagged neurons.
The neurons contained in each image were manually annotated by trained expert annotators.
The dataset is split into a *completely* labeled (all neurons in the image are manually segmented) and a *partly* labeled (a subset of neurons in the image is manually segmented) set.<br>


**How many samples/instances are there in total (of each type, if appropriate)?**

The *completely* labeled set comprises 30 images with 139 labeled neurons in total, and the *partly* labeled set comprises 71 images with 451 labeled neurons in total.<br>


**Does the dataset contain all possible samples or is it a subset (not necessarily random) of samples from a larger set?** *If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).*

The dataset contains a subset of 101 images from the "40x Gen1" set of [[2]](#references).
The full "40x Gen1" set consists of 46,791 images of 4575 different transgenic lines.
From this set, we selected relatively sparse images in terms of number of expressed neurons which seemed feasible for manual annotation.
Thus, our dataset is not representative for the full "40x Gen1" MCFO collection.<br>


**What data does each sample consist of? “Raw” data (e.g., unprocessed text or images) or features? Is there a label or target associated with each sample?** *Please provide a description.*

Each sample consists of a single 3d MCFO image of neurons of the fruit fly.
For each image, we provide a pixel-wise instance segmentation for all separable neurons.
Each sample is stored as a separate *zarr* file ("[zarr](https://zarr.readthedocs.io) is a file storage format for chunked, compressed, N-dimensional arrays based on an open-source specification.").
The image data ("raw") and the segmentation ("gt\_instances") are stored as two arrays within a single zarr file.
The segmentation mask for each neuron is stored in a separate channel.
The order of dimensions is CZYX.
In Python the data can, for instance, be opened with:
```python
import zarr
raw = zarr.open(<path_to_zarr>, path="volumes/raw")
seg = zarr.open(<path_to_zarr>, path="volumes/gt_instances")
```
Zarr arrays are read lazily on-demand.
Many functions that expect numpy arrays also work with zarr arrays.
The arrays can also explicitly be converted to numpy arrays with:
```python
import numpy as np
raw_np = np.array(raw)
```
<br>


**Is any information missing from individual samples?** *If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.*

Not all neuronal structures could be segmented within all images of the provided dataset.
Mainly, there are two reasons: (1) there are overlapping neurons with the same or a similar color that could not be separated due to the partial volume effect, and (2) some neuronal structures cannot be delineated correctly in the presence of noisy background in the same color as the neuron itself.
In the *completely* labeled set all neuronal structures have been segmented, in the *partly* labeled set some structures are missing.<br>


**Are relationships between individual samples made explicit (e.g., users’ movie ratings, social network links)?** *If so, please describe how these relationships are made explicit.*

Yes, one transgenic line is often imaged multiple times as only a stochastic subset of all tagged neurons is visible per MCFO image.
Moreover, the same neuron might be tagged in multiple transgenic lines.<br>


**Are there recommended data splits (e.g., training, development/validation, testing)?** *If so, please provide a description of these splits, explaining the rationale behind them.*

Yes, we provide a recommended data split for training, validation and testing.
The files in the provided download are presorted according to this recommendation.
When splitting the data into sets, we made sure that images of the same transgenic lines are in the same split and paid attention to having similar proportions of images with overlapping neurons as well as having a similar average number of neurons per image in each split.<br>


**Are there any errors, sources of noise, or redundancies in the dataset?** *If so, please provide a description.*

There might be uneven illumination resulting in gaps within neurons in the raw microscopy images as well as the corresponding annotations. This is intrinsic to this kind of light microscopy images.<br>


**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.*

Yes, the dataset is self-contained.
There is an external, additional source of raw images that could potentially be used for self-supervised learning.
The raw images in our dataset are a subset of the released MCFO collection of the FlyLight project [[2]](#references). The whole collection can be downloaded at [https://gen1mcfo.janelia.org](https://gen1mcfo.janelia.org).
Note though that there are no segmentation masks available for these images.<br>


**Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals non-public communications)?** *If so, please provide a description.*

No.<br>


## Collection Process

**How was the data associated with each sample acquired?** *Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.*

The content of the raw images was directly recorded using confocal microscopes.
The annotations were created manually.<br>


**What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *How were these mechanisms or procedures validated?*

Imaging was performed using eight Zeiss LSM 710 or 780 laser scanning confocal microscopes (for more information on the imaging process see [[2]](#references)).
Two trained expert annotators manually segmented and proof-read each other to segment the neurons in these images using the interactive rendering tool [VVD Viewer](https://github.com/JaneliaSciComp/VVDViewer).<br>


**If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**

We manually selected images from the larger "40x Gen1" collection.
We chose images that contained a sparse set of neurons and that contained neurons that preferably were not contained in previously selected images.<br>


**Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**

The data collection process was done by full time employees at the Howard Hughes Medical Institute Janelia Research Campus and the Max-Delbrueck-Center for Molecular Medicine in the Helmholtz Association (MDC).<br>


**Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the samples (e.g., recent crawl of old news articles)?** *If not, please describe the timeframe in which the data associated with the samples was created.*

MCFO selection and manual annotation were mainly done in 2018 and 2019.
The respective acquisition date of the MCFO sample is noted within the sample name in "YYYYMMDD" format. Most samples of our dataset were acquired in 2017 and 2018.<br>


**Were any ethical review processes conducted (e.g., by an institutional review board)?** *If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.*

There was no ethical review process conducted as we did not record any new animal data, the dataset does not relate to people and it does not contain confidential data.<br>


**Does the dataset relate to people?** *If not, you may skip the remaining questions in this section.*

No<br>


## Preprocessing/cleaning/labeling

**Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *If so, please provide a description. If not, you may skip the remainder of the questions in this section.*

The following preprocessing was done for each image:
The central brain and part of the ventral nerve cord (VNC) were recorded in tiles by the light microscope.
The tiles were stitched together and distortion corrected (for more information see [[5]](#references)).<br>


**Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *If so, please provide a link or other access point to the “raw” data.*

The original images are available at [https://gen1mcfo.janelia.org](https://gen1mcfo.janelia.org).<br>


**Is the software used to preprocess/clean/label the samples available?** *If so, please provide a link or other access point.*

The image processing, such as distortion correction and stitching, is done by using the open-source software Janelia Workstation [[6]](#references).<br>


## Uses

**Has the dataset been used for any tasks already?** *If so, please provide a description.*

In [[7]](#references), an earlier, unpublished version of our dataset has been used to qualitatively evaluate *PatchPerPix*, a deep learning-based instance segmentation method.
The trained model was then applied to $\sim$40.000 samples of the MCFO collection [[2,8]](#references) to search for given neuronal structures extracted from electron microscopy (EM) data [[9]](#references).
*PatchPerPix* is also used as a baseline to showcase this published version of our dataset.<br>


**Is there a repository that links to any or all papers or systems that use the dataset?** *If so, please provide a link or other access point.*

As they are getting published, we will reference them at [https://kainmueller-lab.github.io/fisbe](https://kainmueller-lab.github.io/fisbe).<br>


**What (other) tasks could the dataset be used for?**

The dataset can be used for a wide range of method development tasks such as capturing long-range dependencies, segmentation of thin filamentous structures, self- and semi-supervised training or denoising.
Advances in these areas can in turn facilitate scientific discoveries in basic neuroscience by providing improved neuron reconstructions for morphological and functional analyses.<br>


## Distribution

**Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *If so, please provide a description.*

The dataset will be publicly available.<br>


**How will the dataset be distributed (e.g., tarball on website, API, GitHub)** *Does the dataset have a digital object identifier (DOI)?*

The dataset will be distributed through [zenodo](https://zenodo.org) (DOI: [10.5281/zenodo.10875063](https://zenodo.org/doi/10.5281/zenodo.10875063)) and our project page [https://kainmueller-lab.github.io/fisbe](https://kainmueller-lab.github.io/fisbe).<br>


**When will the dataset be distributed?**

With publication of the accompanying paper.<br>


**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.*

The dataset will be distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).<br>


**Have any third parties imposed IP-based or other restrictions on the data associated with the samples?** *If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.*

All MCFO images have previously been made publicly available by [[2]](#references) under the same license (CC BY 4.0) at [https://gen1mcfo.janelia.org](https://gen1mcfo.janelia.org).<br>


## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**

Lisa Mais supports and maintains the dataset.<br>


**How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**

Lisa Mais and Dagmar Kainmueller can be contacted at \{firstname.lastname\}@mdc-berlin.de.<br>


**Is there an erratum?** *If so, please provide a link or other access point.*

Errata will be published at [https://kainmueller-lab.github.io/fisbe](https://kainmueller-lab.github.io/fisbe).<br>


**Will the dataset be updated (e.g., to correct labeling errors, add new samples, delete samples)?** *If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?*

The dataset will be updated to correct erroneous segmentation and potentially to add new samples and annotations.
It will be updated when a relevant number of updates has accumulated.
Updates will be communicated through [https://kainmueller-lab.github.io/fisbe](https://kainmueller-lab.github.io/fisbe).<br>


**Will older versions of the dataset continue to be supported/hosted/maintained?** *If so, please describe how. If not, please describe how its obsolescence will be communicated to users.*

We publish our dataset on zenodo.
Zenodo supports versioning, including DOI versioning.
Older versions of the dataset will thus stay available.<br>

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.*

We welcome contributions to our dataset.
Errata, new samples and annotations and other contributions can be contributed via *github issues* at [https://kainmueller-lab.github.io/fisbe](https://kainmueller-lab.github.io/fisbe).
We will verify such contributions and update the dataset accordingly.<br>


## References

[1] Gebru et al.: *Datasheets for Datasets* [arxiv:1803.09010](https://arxiv.org/abs/1803.09010)<br>
[2] Meissner et al.: *A searchable image resource of Drosophila GAL4 driver expression patterns with single neuron resolution*, [10.7554/eLife.80660](https://doi.org/10.7554/eLife.80660)<br>
[3] Nern et al.: *Optimized tools for multicolor stochastic labeling reveal diverse stereotyped cell arrangements in the fly visual system*, [10.1073/pnas.1506763112](https://www.pnas.org/doi/10.1073/pnas.1506763112)<br>
[4] Jenett et al.: *A GAL4-driver line resource for Drosophila neurobiology*, [10.1016/j.celrep.2012.09.011](https://pubmed.ncbi.nlm.nih.gov/23063364/)<br>
[5] Yu and Peng: *Automated high speed stitching of large 3D microscopic images*, [10.1109/ISBI.2011.5872396](https://ieeexplore.ieee.org/document/5872396)<br>
[6] Rokicki et al.: *Janelia Workstation Codebase*, [10.25378/janelia.8182256.v1](https://janelia.figshare.com/articles/software/Janelia_Workstation_Codebase/8182256)<br>
[7] Mais and Hirsch et al.: *PatchPerPix for Instance Segmentation*, [10.1007/978-3-030-58595-2_18](https://github.com/Kainmueller-Lab/PatchPerPix)<br>
[8] Mais et al.: *PatchPerPixMatch for automated 3D search of neuronal morphologies in light microscopy*, [10.1101/2021.07.23.453511](https://www.biorxiv.org/content/10.1101/2021.07.23.453511v1)<br>
[9] Scheffer et al.: *A connectome and analysis of the adult Drosophila central brain*, [10.7554/eLife.57443](https://elifesciences.org/articles/57443)<br>
