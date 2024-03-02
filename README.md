# HTINet2
A freamwork for herbal drug target prediction.

## Environment Requirement
* torch == 2.0.1
* numpy == 1.25.2
* pandas == 1.5.3


## Usage
### component 1
The framework we built is not an end-to-end model, component 1 and the subsequent ones are separate. Component 1 utilizes various graph representation learning methods to represent the KG's entities, and the embeddings data is partially available in `/component-1/embedding`.

### component 2&3
#### train

```
cd code
python train.py
```

#### test
```
cd code
python test.py
```