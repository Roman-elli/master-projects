#!/bin/bash
# Unlike Tutorial 2, we don't train inside the container — you already
# trained your model in the notebook. This just starts the API, and it
# will fail fast with a clear error if model.pkl isn't next to it.
#!/bin/bash
cd src
python app.py
