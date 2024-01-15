# xcube-ogccov: OGC API - Coverages data store for xcube

This data store allows xcube to read data via OGC API - Coverages.

xcube-ogccov is an
[xcube data store](https://xcube.readthedocs.io/en/latest/dataaccess.html#data-store-framework)
which allows xcube to read data from a server implementing the
[OGC API - Coverages](https://docs.ogc.org/DRAFTS/19-087.html)
API. The data store maps its data opener parameters to OGC API parameters,
fetches a NetCDF of the requested coverage, and makes it available to the
caller as an xcube dataset.

At the time of writing, OGC API - Coverages is still in a draft state, so
compatibility with all implementing servers cannot be guaranteed. xcube-ogccov
has mainly been tested with xcube’s own  OGC API - Coverage server
implementation.

## Installation

1. Clone the github repository
2. Set the current directory to the root directory of the repository
3. Use conda or an equivalent tool (we recommend
   [mamba](https://mamba.readthedocs.io/)) to create a new Python environment
   with the required dependencies
4. Run the `setup.py` script to install the package

```bash
git clone https://github.com/dcs4cop/xcube-ogccov.git
cd xcube-ogccov
mamba env create
mamba activate xcube-ogccov
python setup.py install
```

## Usage

TODO
