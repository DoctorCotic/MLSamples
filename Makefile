# ISTM_IMAGE = "sonm/istm-nn"
# ISTM_DOCKERFILE = "istm.Dockerfile"

BINNARYCROSS = "binnarycross"
MNIST = "mnist"
ISS = "iss"
CONV = "conv"
VGG16 = "vgg16"
TEST = "test"

mnist:
	docker build -t sonm/${MNIST}-nn:latest -f ${MNIST}.Dockerfile .

binnarycross:
	docker build -t sonm/${BINNARYCROSS}-nn:latest -f ${BINNARYCROSS}.Dockerfile .

iss:
	docker build -t sonm/${ISS}-nn:latest -f ${ISS}.Dockerfile .

conv:
	docker build -t sonm/${CONV}-nn:latest -f ${CONV}.Dockerfile .

vgg16:
	docker build -t sonm/${VGG16}-nn:latest -f ${VGG16}.Dockerfile .

test:
	docker build -t sonm/${TEST}-nn:latest -f ${TEST}.Dockerfile .



