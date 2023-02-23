#!/bin/bash

# ovftoolでESXiにデプロイ
#

OVFTOOL="/Applications/VMware\ OVF\ Tool/ovftool"
NAME="vmware5-4"
DATASTORE="datastore1-4"
IMAGE="~/Documents/Forse/vmware/vmware3-cent9/vmware3-cent9.ovf"
DST="vi://root:Me1onpan#@192.168.210.174/"

${OVFTOOL} \
		-ds=${DATASTORE} \
		-dm=thick \
		-nw="VM Network" \
		-n=${NAME} \
		${IMAGE} \
		${DST}
