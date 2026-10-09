// Facts the pages share.

export const repository = 'https://github.com/AshishKumar4/sparx';
export const install = 'pip install git+https://github.com/AshishKumar4/sparx';
export const colab = (notebook) => `https://colab.research.google.com/github/AshishKumar4/sparx/blob/main/site/notebooks/${notebook}.ipynb`;
export const source = (path, line) => `${repository}/blob/main/${path}${line ? `#L${line}` : ''}`;
