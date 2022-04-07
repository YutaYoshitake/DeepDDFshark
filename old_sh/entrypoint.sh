# a temporary file on tmpfs (ramdisk)
ID_RSA=/ram/id_rsa

# mandatory
SSHFS_OPT="-o follow_symlinks,reconnect,allow_other,idmap=user"
# for convenience
SSHFS_OPT="${SSHFS_OPT},StrictHostKeyChecking=no,ServerAliveInterval=60"
# for better performance
SSHFS_OPT="${SSHFS_OPT},auto_cache,kernel_cache,large_read,big_writes,compression=no,no_remote_lock"

# import sshkey from SSH_KEY env
echo -n "${SSH_KEY}" | tr '_' '\n' > ${ID_RSA}
chmod 600 ${ID_RSA}

# sshfs mount
sshfs -o IdentityFile=${ID_RSA} ${SSHFS_OPT} ${HOST_USER}@${HOST_IP}:${CWD} /app

# remove private key. If you need to remount, comment out this line.
# rm ${ID_RSA} # なくなるかも？

# start
cd /app/project
# export PS1='\[\033[01;35m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
# cd /app && rm -r build && mkdir build && cd build && cmake -DCMAKE_CXX_STANDARD=17 .. && make -j && cd ..
export PS1='\[\033[01;35m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
bash
