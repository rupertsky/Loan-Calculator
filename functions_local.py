import pandas as pd


def error_back(message):
    print(message)


def import_csv(csv_file,  delim=',', enc='utf-8'):
    """
    Funcion para evaluar si el archivo introducido existe y
    es de tipo csv, devuelve un archivo DataFrame.
    :param csv_file: Archivo csv.
    :return:Pandas DataFrame si es True,
            de lo contrario False.
    """
    try:
        return pd.read_csv(csv_file, encoding=enc, delimiter=delim), 1
    except UnicodeDecodeError as e:
        error_back(f'Error de encode al intentar importar csv con {enc}')
        return None, 2
    except FileNotFoundError as e:
        error_back(str(e))
        return None, 3
    except Exception as e:
        error_back(str(e))
        return None, 4


def import_csv_front(archivo, df_container, delim, encode='utf-8'):
    """
    Esta funcion recibe el nombre del archivo, un almacenador, un delimitador,
    y un encoding, devolviendo un DataFrame si no encuentra errores.
    :param archivo:
    :param df_container:
    :param delim:
    :param encode:
    :return: DataFrame
    """
    df_c, code = import_csv(archivo, delim, encode)
    add_container = False
    if code == 3:
        print('Archivo inexistente y/o diferente a .csv!')
    elif code == 2:
        enc = input('ingrese el encode correcto: ')
        return import_csv_front(archivo, df_container, delim, encode=enc)
    elif code == 4:
        print('Error inesperado')
    else:
        add_container = True
    if add_container:
        df_container.append(df_c)
        return 1
    else:
        return 0


def dfs_list():
    """
    Funcion que devuelve una lista conteniendo un numero determinado de
    dataframes.
    :return: Lista de DataFrames.
    """
    df_container = []
    while True:
        try:
            continuar = True
            arc_cargados = 1
            while continuar:
                agr_archivo = int(input("""Desea agregar un nuevo archivo?
                1. Si
                2. No\n"""))
                if agr_archivo == 1:
                    archivo = input(f'Ingrese la ruta del archivo {arc_cargados}: ')
                    delim = input('Ingrese el delimiter: ')
                    arc_cargados += import_csv_front(archivo, df_container, delim)
                elif agr_archivo == 2:
                    continuar = False
                else:
                    print('Opcion no valida, favor ingresar 1 o 2')
        except ValueError:
            print('Debe ingresar un valor numerico!')
            continue
        return df_container


def main():
    return dfs_list()


if __name__ == "__main__":
    main()
