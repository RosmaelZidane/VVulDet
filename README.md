## Domain-Aware Graph Neural Networks for Source Code Vulnerability Detection

### Experiment replication.

1. Clone the project repository.
   ```bash
   git clone https://github.com/RosmaelZidane/VVulDet.git
   ```
2.  Install the required Python packages.
   ``` ash
pip install -r requirements
```
3. Dataset and CPG extraction.
   - Dataset: We used publicly available datasets named BigVul, ProjectKB, MegaVul, and CVEFixes.
   - CPG extraction: We use Joern to parse the source code, extracting relevant nodes and edge data.

Running the following commands will install a specific Joern version for CPG extraction and download from our drive a Python version of the CVEFixes dataset.
```bash
chmod +x ./run.sh
./run.sh
```
+
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


