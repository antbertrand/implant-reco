# gcaesthetics-implantbox
A project that scans a serial number on a breast implant


## Interesting entry points

main.py : complete app with detection until text

eurosilicone-reader.py : acquisition app (saves chips to /var/eurosilicone/acqusitions)

uploader.py : will upload acquisitions to Azure (use this in a CRON)

