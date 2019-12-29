sudo apt-get install build-essential
cc usbreset.c -o usbreset
sudo mv usbreset /usr/local/bin/
sudo chmod +x /usr/local/bin/usbreset
