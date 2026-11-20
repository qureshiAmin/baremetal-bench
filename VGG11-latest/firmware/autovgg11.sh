#!/bin/bash
riscv64-unknown-elf-objcopy -O binary vgg11_rv_demo_vc707.elf vgg11_rv_demo_vc707.bin
sudo dd if=vgg11_rv_demo_vc707.bin of=/dev/sdd bs=512 seek=34 conv=fsync
