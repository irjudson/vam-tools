#!/bin/bash
USER_NAME="irjudson"  # change this
MOUNT_POINT="/home/$USER_NAME/iPhone"

mkdir -p "$MOUNT_POINT"

# Wait for device to be ready
for i in {1..10}; do
    if ideviceinfo &>/dev/null; then
        break
    fi
    sleep 1
done

if ! mountpoint -q "$MOUNT_POINT"; then
    sudo -u "$USER_NAME" ifuse "$MOUNT_POINT" && \
        echo "iPhone mounted at $MOUNT_POINT"
fi
