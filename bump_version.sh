source VERSION
sed -e 's#{VERSION}#'"${VERSION}"'#g' setup_template.cfg > setup.cfg

rm -R dist
rm -R build
python3 setup.py build sdist bdist_wheel

git add --all
git commit -m "Building a new version ${VERSION}"
git tag -a v${VERSION} -m "Building a new version ${VERSION}"
git push
git push origin v${VERSION}
