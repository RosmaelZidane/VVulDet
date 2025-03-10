## Domain-Aware Graph Neural Networks for Source Code Vulnerability Detection
Paper submited to [JSS](https://www.sciencedirect.com/journal/journal-of-systems-and-software) Journal.

### Section1: Experiment replication: The source code for training the model is located in ./sourcescripts, and the dataset construction instructions can be found in ./domainknowledge.

1. Clone the project repository.
   ```bash
   git clone https://github.com/RosmaelZidane/VVulDet.git
   ```
2.  Install the required Python packages.
   ``` ash
pip install -r requirements
```
3. Dataset and CPG extraction.
   - Dataset: We used publicly available datasets named [BigVul-C/C++](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view), [Project_KB-Java](https://github.com/SAP/project-kb.git), [MegaVul-Java](https://github.com/Icyrockton/MegaVul), and [CVEFixes-Python](https://github.com/secureIT-project/CVEfixes).
   - CPG extraction: We use [Joern](https://joern.io/) to parse the source code, extracting relevant nodes and edge data.

Running the following commands will install a specific Joern version for CPG extraction and download from our drive a Python version of the CVEFixes dataset.
```bash
chmod +x ./run.sh
./run.sh
```
```bash
./zrun/getjoern.sh
```
4. Data processing
```bash
./zrun/dataprocessing.sh
```
5. Training the main model: without domain knowledge.

Training the main model takes considerable time since the CodeBERT model is fine-tuned for embedding. Moreover, generating contextualized graph embedding with the Node2vec model also takes some time and is applicable to every model's version.
```bash
./zrun/trainmain.sh
```
6. Training with Domain Knowledge
   - CVE Description
```bash
./zrun/trainwithcve.sh
```
   - CWE Description
```bash
.zrun/traimwithcwe.sh
```
   - CVE and CWE Description
```bash
./zrun/trainwithcvecwe.sh
```
   - Reference functions
```bash
./zrun/trainwithsample.sh
```

### Section 2: Domain Knowledge Collection

- The dataset used are available in ./domainknowledge as well as instruction to add domain information in a given dataset.


### Citation

To be provided




