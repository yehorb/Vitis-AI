echo "Install all the python dependencies using pip"
pip install --trusted-host xcdpython.xilinx.com -r requirements_ptq.txt

ROOT_DIR=${PWD}
cd ${ROOT_DIR}/code
pip install -v -e .
cd ${ROOT_DIR}
