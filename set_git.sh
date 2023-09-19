#!/bin/bash
# git config --list check the info

function usage {
  echo "Usage: $0 [-h | -l | -n]"
  echo "  -h: set git user to guanbinhuang"
  echo "  -l: set git user to coeyliang"
  echo "  -n: unset git user"
  exit 1
}

if [ $# -eq 0 ]
then
  # none
  git config --global user.name ""
  git config --global user.email ""
  exit 0
fi

while getopts "hln" option
do
  case "${option}" in
    h)
      # hgb
      git config --global user.name "guanbinhuang"
      git config --global user.email "qsmy122011@gmail.com"
      ;;
    l)
      # coey
      git config --global user.name "coeyliang"
      git config --global user.email "coeyliang20@gmail.com"
      ;;
    n)
      # none
      git config --global user.name ""
      git config --global user.email ""
      ;;
    *)
      usage
      ;;
  esac
done
