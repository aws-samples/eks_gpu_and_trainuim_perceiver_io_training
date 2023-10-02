#!/bin/bash -x

cat << EOF | tee --append /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF

GRUB_CMDLINE_LINUX="rdblacklist=nouveau"
update-grub
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
aws s3 ls --recursive s3://ec2-linux-nvidia-drivers/
chmod +x NVIDIA-Linux-x86_64*.run
/bin/sh ./NVIDIA-Linux-x86_64*.run
nvidia-smi -q | head
touch /etc/modprobe.d/nvidia.conf
echo "options nvidia NVreg_EnableGpuFirmware=0" | tee --append /etc/modprobe.d/nvidia.conf
apt-get install -y lightdm ubuntu-desktop
