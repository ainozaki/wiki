#pragma once

#include "my_buf.h"

struct net_device;

struct net_device_ops{
    int (*transmit)(net_device *dev, my_buf* buf);
    int (*poll)(net_device *dev);
};

struct net_device{
    char ifname[32]; // インターフェース名
    uint8_t mac_address[6];
    net_device_ops ops;
    // ip_device* ip_dev; // 第3章で使用
    net_device* next;
    uint8_t data[];
};
