export VERSION_NUMBER=$(echo $RID | cut -d'v' -f 2).$TRAVIS_BUILD_NUMBER
sed -i -e 's/0\.0\.0/'"$VERSION_NUMBER"'/g' setup.py
