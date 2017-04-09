#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

# 对比是否使用列表解析
constant = 10**3
begin1 = time.time()
d = { k : [0 for i in xrange(constant)] for k in xrange(constant)}
end1 = time.time()

begin2 = time.time()
d = {}
for k in xrange(constant):
    for i in xrange(constant):
        if i == 0:
            d[k] = []
        else:
            d[k].append(0)
end2 = time.time()

print ("使用列表解析 %f 和未使用列表解析 %f" % ((end1 - begin1),(end2 - begin2)))

