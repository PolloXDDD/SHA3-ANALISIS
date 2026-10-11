# SHA3-ANALISIS

Estudio reproducible de criptoanálisis de SHA-3 propuesto por **Kaoru Aguilera Katayama**: números hipercomplejos, metacadenas de Markov autorreferenciales con estructura diádica finita, Tarjan/2-SAT, intervenciones causales y métricas de desplazamiento.

**[Paper en inglés (PDF)](paper/main.pdf)** · **[LaTeX](paper/main.tex)** · **[Bibliografía](paper/references.bib)** · **[Protocolo y alcance](docs/METHODOLOGY.md)**

## Resultados principales

| Ruta | Implementación y resultado |
|---|---|
| Hipercomplejos | Representación exacta de cada lane en el álgebra de Clifford real `Cl(0,6)`; producto geométrico implementado. La distancia de coeficientes al cuadrado equivale a cuatro veces la distancia de Hamming. |
| Diferenciales | Familia universal de colisiones para SHA3-256 reducido a **una ronda inicial**. Certificado y demostración incluidos; se explota la estructura conocida de paridad de columnas. |
| Difusión | Con la diferencia elegida, la media de bits cambiados del estado es **4.009, 82.113, 692.113 y 799.902** tras 1, 2, 3 y 4 rondas. Se ejecutaron 8,192 pares por diferencia. |
| Metacadenas | Predictor finito con seis resoluciones, fase de ronda y realimentación de su propia predicción. Mejora frente a usar solo la fase; la variante sin autorreferencia predice ligeramente mejor. |
| Tarjan y 2-SAT | SCC de grafos estructurales, intervención y ciclos proyectados; solver 2-SAT exacto. Demostración de por qué la relación local de χ no admite una codificación 2-CNF exacta, incluso con auxiliares existenciales. |
| Búsqueda de objetivos de 12 bits | En dos rondas: **38/48** aciertos del modelo completo frente a **21/48** aleatorios. Repetición con semillas nuevas: **34/48** frente a **22/48**. |
| Controles a 24 rondas | En la repetición, búsqueda guiada y aleatoria lograron **21/48** cada una, siempre sobre objetivos truncados a 12 bits. No se obtuvo un ataque de colisión o preimagen sobre la salida completa de SHA3-256. |

Las ablaciones no establecen una contribución independiente de la autorreferencia o de los SCC al éxito de búsqueda. El paper conserva resultados favorables y desfavorables, intervalos de confianza y correcciones por comparaciones múltiples. La colisión de una ronda es una construcción reproducible basada en una estructura conocida, no una afirmación de novedad ni de ruptura del SHA-3 completo.

## Reproducir

Python 3.12 recomendado. El núcleo usa NumPy; los experimentos y figuras también usan SciPy y Matplotlib.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m sha3lab.experiments --out results
python -m sha3lab.ablations --out results
python scripts/verify_results.py
python scripts/build_report.py
python scripts/build_pdf.py
```

Para recrear las versiones exactas de las dependencias usadas, instala `requirements-repro.txt`. La compilación requiere una distribución LaTeX con `pdflatex`, BibTeX y los paquetes declarados en `paper/main.tex`. El script compila en un directorio temporal y reemplaza el PDF final cuando está completo.

Prueba rápida de ejecución, con un directorio separado:

```bash
python -m sha3lab.experiments --quick --out results-quick
```

Los datos rápidos no son los del paper. Los tiempos de ejecución del entorno original se guardan en los manifiestos; no representan benchmarks entre métodos.

## Convenciones que importan

- **Rondas reducidas:** primeras `r` rondas, constantes `RC[0:r]`. No se confunden con las últimas `r` rondas de Keccak-p. Con 24 rondas, la función es SHA3-256 normal.
- **Bits:** índice de lane `x + 5*y`, bits little endian dentro de cada lane.
- **Objetivo de 12 bits:** primer byte del digest más los cuatro bits bajos del segundo byte. Una preimagen de esta proyección no es una preimagen de 256 bits.
- **Ciclos exhaustivos:** se estudia una función de 12 bits construida proyectando y reinyectando una permutación Keccak de 25 bits. La definición exacta está en el paper.
- **Causalidad:** intervenciones pareadas sobre bits en un mecanismo conocido; la correlación de Pearson se informa por separado.
- **Autoría asistida:** el autor propuso las direcciones; ChatGPT asistió con la formalización, código, experimentos y redacción.

## Archivos

| Ruta | Contenido |
|---|---|
| `sha3lab/keccak.py` | Implementaciones escalar y por lotes, padding y SHA-3 completo. |
| `sha3lab/hypercomplex.py` | Clifford, transformación exacta y características de grados. |
| `sha3lab/metamarkov.py` | Predictor autorreferencial finito y controles. |
| `sha3lab/graphs.py` | Tarjan iterativo, 2-SAT, dependencias y análisis exhaustivo de χ. |
| `sha3lab/attacks.py` | Colisión constructiva y búsqueda con presupuesto. |
| `sha3lab/experiments.py` | Experimento principal determinista. |
| `sha3lab/ablations.py` | Repetición independiente y ablaciones. |
| `tests/` | 15 pruebas funcionales/matemáticas con comparaciones independientes. |
| `results/` | Datos crudos, resúmenes, semillas, 768 endpoints verificables y huellas del código. |
| `scripts/` | Verificación independiente y generación de figuras/tablas. |
| `paper/` | LaTeX, BibTeX, tablas generadas, figuras vectoriales y PDF. |

Todas las afirmaciones numéricas deben interpretarse con el objeto, número de rondas, anchura y presupuesto indicados. Las funciones de este repositorio son herramientas de investigación.
