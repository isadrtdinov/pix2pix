#!/bin/bash

mkdir -p 'datasets'

if [ $1 == 'facades' ]
then
  wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
  tar -xf facades.tar.gz
  rm facades.tar.gz
  mv ./facades/ datasets/facades/

elif [ $1 == 'maps' ]
then
  wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
  tar -xf maps.tar.gz
  rm maps.tar.gz
  mv ./maps/ datasets/maps/

elif [ $1 == 'edges2shoes' ]
then
  wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz
  tar -xf edges2shoes.tar.gz
  rm edges2shoes.tar.gz
  mv ./edges2shoes/ datasets/edges2shoes/

elif [ $1 == 'night2day' ]
then
  wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/night2day.tar.gz
  tar -xf night2day.tar.gz
  rm night2day.tar.gz
  mv ./night2day/ datasets/night2day/

elif [ $1 == 'churches' ]
then
  wget https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/EW8Jcmbzfg1big -O churches-v2.tar.gz
  tar -xf churches-v2.tar.gz
  rm churches-v2.tar.gz
  mv ./churches/ datasets/churches/
fi

