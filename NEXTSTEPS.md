

# Next Steps for ndx-pose Extension for NWB

## Creating Your Extension

1. In a terminal, change directory into the new ndx-pose directory.

2. Add any packages required by your extension to `requirements.txt` and `setup.py`.

3. Run `python -m pip install -r requirements.txt` to install the `pynwb` package
and any other packages required by your extension.

4. Modify `src/spec/create_extension_spec.py` to define your extension.

5. Run `python src/spec/create_extension_spec.py` to generate the
`spec/ndx-pose.namespace.yaml` and
`spec/ndx-pose.extensions.yaml` files.

6. Define API classes for your new extension data types.

    - As a starting point, `src/pynwb/__init__.py` includes an example for how to use
      the `pynwb.get_class` to get a basic Python class for your new extension data
      type. This class contains a constructor and properties for the new data type.
    - Instead of using `pynwb.get_class`, you can define your own custom class for the
      new type, which will allow you to customize the class methods, customize the
      object mapping, and create convenience functions. See
      [https://pynwb.readthedocs.io/en/stable/tutorials/general/extensions.html](https://pynwb.readthedocs.io/en/stable/tutorials/general/extensions.html)
      for more details.

7. Define tests for your new extension data types in `src/pynwb/tests` or `src/matnwb/tests`.
A test for the example `TetrodeSeries` data type is provided as a reference and should be
replaced or removed.

     - Python tests should be runnable by executing [`pytest`](https://docs.pytest.org/en/latest/)
     from the root of the extension directory. Use of PyNWB testing infrastructure from
     `pynwb.testing` is encouraged (see
    [documentation](https://pynwb.readthedocs.io/en/stable/pynwb.testing.html)).
     - Creating both **unit tests** (e.g., testing initialization of new data type classes and
     new functions) and **integration tests** (e.g., write the new data types to file, read
     the file, and confirm the read data types are equal to the written data types) is
     highly encouraged.

8. You may need to modify `setup.py` and re-run `python setup.py install` if you
use any dependencies.


## Documenting and Publishing Your Extension to the Community

1. Install the latest release of hdmf_docutils: `python -m pip install hdmf-docutils`

2. Start a git repository for your extension directory ndx-pose
 and push it to GitHub. You will need a GitHub account.
    - Follow these directions:
  https://help.github.com/en/articles/adding-an-existing-project-to-github-using-the-command-line

3. Change directory into `docs`.

4. Run `make html` to generate documentation for your extension based on the YAML files.

5. Read `docs/README.md` for instructions on how to customize documentation for
your extension.

6. Modify `README.md` to describe this extension for interested developers.

7. Add a license file. Permissive licenses should be used if possible. **A [BSD license](https://opensource.org/licenses/BSD-3-Clause) is recommended.**

8. Make a release for the extension on GitHub with the version number specified. e.g. if version is 0.1.0, then this page should exist: https://github.com/rly/ndx-pose/releases/tag/0.1.0 . For instructions on how to make a release on GitHub see [here](https://help.github.com/en/github/administering-a-repository/creating-releases).

9. Publish your updated extension on [PyPI](https://pypi.org/).
    - Follow these directions: https://packaging.python.org/tutorials/packaging-projects/
    - You may need to modify `setup.py`
    - If your extension version is 0.1.0, then this page should exist: https://pypi.org/project/ndx-pose/0.1.0

   Once your GitHub release and ``setup.py`` are ready, publishing on PyPI:
    ```bash
    python setup.py sdist bdist_wheel
    twine upload dist/*
    ```

10. Go to https://github.com/nwb-extensions/staged-extensions and fork the
repository.

11. Clone the fork onto your local filesystem.

12. Copy the directory `staged-extensions/example` to a new directory
`staged-extensions/ndx-pose`:

    ```bash
    cp -r staged-extensions/example staged-extensions/ndx-pose
    ```

13. Edit `staged-extensions/ndx-pose/ndx-meta.yaml`
with information on where to find your NWB extension.
    - The YAML file MUST contain a dict with the following keys:
      - name: extension namespace name
      - version: extension version
      - src: URL for the main page of the public repository (e.g. on GitHub, BitBucket, GitLab) that contains the sources of the extension
      - pip: URL for the main page of the extension on PyPI
      - license: name of the license of the extension
      - maintainers: list of GitHub
      usernames of those who will reliably maintain the extension
    -

  You may copy and modify the following YAML that was auto-generated:
```yaml
name: ndx-pose
version: 0.1.0
src: https://github.com/rly/ndx-pose
pip: https://pypi.org/project/ndx-pose/
license: BSD 3-Clause
maintainers: 
  - rly
  - bendichter
  - AlexEMG
```

14. Edit `staged-extensions/ndx-pose/README.md`
to add information about your extension. You may copy it from
`ndx-pose/README.md`.

```bash
cp ndx-pose/README.md staged-extensions/ndx-pose/README.md
```

15. Add and commit your changes to Git and push your changes to GitHub.
```
cd staged-extensions
git add ndx-pose
git commit -m "Add new catalog entry for ndx-pose" .
git push
```

16. Open a pull request. Building of your extension will be tested on Windows,
Mac, and Linux. The technical team will review your extension shortly after
and provide feedback and request changes, if any.

17. When your pull request is merged, a new repository, called
ndx-pose-feedstock will be created in the nwb-extensions
GitHub organization and you will be added as a maintainer for that repository.


## Updating Your Published Extension

1. Update your ndx-pose GitHub repository.

2. Publish your updated extension on PyPI.

3. Fork the ndx-pose-feedstock repository on GitHub.

4. Open a pull request to test the changes automatically. The technical team
will review your changes shortly after and provide feedback and request changes,
 if any.

5. Your updated extension is approved.
