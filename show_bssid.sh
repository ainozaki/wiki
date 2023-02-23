#!/bin/bash

# bssidを1秒毎に表示するスクリプト
#
# $ sh ./show_bssid.sh
# =====================
# Show SSID and BSSID
# =====================
# codeblue
# codeblue
# codeblue
# codeblue
# ^C
#

index=0

show_ssid(){
	AIRPORT='/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport'
	RET=$(${AIRPORT} -I)
	echo "${RET}" | awk '$1=="SSID:" {printf "%s\t", $2 }'
	echo "${RET}" | awk '$1=="BSSID:" {printf "%s\n", $2 }'
	index+=1
}

echo "====================="
echo "Show SSID and BSSID"
echo "====================="
while true
do
	show_ssid
	sleep 1
done
