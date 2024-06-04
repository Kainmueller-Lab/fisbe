## FISBe: A real-world benchmark dataset for instance segmentation of long-range thin filamentous structures

This is the repository for our FISBe project page at https://kainmueller-lab.github.io/fisbe/ .

### How to test github page locally

- setup conda env
```
  conda install -n base conda-libmamba-solver
  conda config --set solver libmamba
  conda create --name gh-pages python
  conda activate gh-pages
  conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9 compilers
  conda install -c conda-forge ruby
  gem install jekyll bundler
```
- go to folder
```
  bundle install
  bundle exec jekyll serve
```
- maybe
```
  ln -s ~/anaconda3/envs/gh-pages/bin/ruby ~/anaconda3/envs/gh-pages/share/rubygems/bin/ruby
``` 
