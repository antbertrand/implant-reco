#!/bin/bash
cstate=$(netstat -na | grep "tcp" | grep "54.228.200.242:22" | tr -s " " | cut -d " " -f 6 | head -n 1)
cdate=$(date "+%Y-%m-%d %H:%M:%S")
if [[ $cstate == "ESTABLISHED" ]] || [[ $cstate == "TIME_WAIT" ]]
then
    echo "[$cdate] SSH connection up. ($cstate)" >> /var/log/check-ssh-relay.log
else
    echo "[$cdate] SSH connection broken. ($cstate) Reconnecting..." >> /var/log/check-ssh-relay.log
    ssh -N -f -R 12123:localhost:22 pjgrizel@redmine.numericube.com
fi

