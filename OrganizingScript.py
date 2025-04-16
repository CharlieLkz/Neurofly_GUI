import os
import shutil
import re
import time
import errno

def main():
    # Ruta base del proyecto
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas de las carpetas
    filtered_data_dir = os.path.join(base_path, "FilteredData")
    organized_data_dir = os.path.join(base_path, "OrganizedData")
    
    # Manejar la eliminación de OrganizedData con mejor control de errores
    if os.path.exists(organized_data_dir):
        print(f"Eliminando carpeta existente: {organized_data_dir}")
        try:
            # Intenta eliminar la carpeta
            shutil.rmtree(organized_data_dir)
        except PermissionError:
            print("Error de permisos al eliminar la carpeta. Intentando método alternativo...")
            
            # Intenta vaciar la carpeta archivo por archivo
            for root, dirs, files in os.walk(organized_data_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        print(f"No se pudo eliminar el archivo: {os.path.join(root, name)}")
                
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except:
                        print(f"No se pudo eliminar la carpeta: {os.path.join(root, name)}")
            
            print("Continuando con el proceso usando la carpeta existente...")
        except Exception as e:
            print(f"Error al eliminar carpeta: {str(e)}")
            print("Continuando con el proceso usando la carpeta existente...")
    
    # Asegurarse de que exista la carpeta OrganizedData
    if not os.path.exists(organized_data_dir):
        print(f"Creando carpeta: {organized_data_dir}")
        os.makedirs(organized_data_dir)
    
    # Diccionario para almacenar archivos por categoría
    categorias = {}
    
    # Recorrer todas las subcarpetas en FilteredData
    print("Buscando archivos CSV en FilteredData...")
    for root, dirs, files in os.walk(filtered_data_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                # Clasificar el archivo según su nombre
                categoria = clasificar_archivo(file)
                
                if categoria:
                    if categoria not in categorias:
                        categorias[categoria] = []
                    
                    categorias[categoria].append(file_path)
                    print(f"Encontrado: {file} → categoría: {categoria}")
    
    # Crear subcarpetas y copiar archivos
    for categoria, archivos in categorias.items():
        # Crear carpeta de categoría si no existe
        categoria_dir = os.path.join(organized_data_dir, categoria)
        if not os.path.exists(categoria_dir):
            os.makedirs(categoria_dir)
            print(f"\nCreando carpeta para categoría: {categoria}")
        else:
            print(f"\nUsando carpeta existente para categoría: {categoria}")
        
        # Copiar archivos a la carpeta correspondiente
        for archivo in archivos:
            nombre_archivo = os.path.basename(archivo)
            destino = os.path.join(categoria_dir, nombre_archivo)
            
            try:
                print(f"  Copiando: {nombre_archivo}")
                # Si el archivo ya existe, intenta eliminarlo primero
                if os.path.exists(destino):
                    try:
                        os.remove(destino)
                    except:
                        print(f"  No se pudo eliminar archivo existente: {destino}")
                        continue
                
                shutil.copy2(archivo, destino)  # copy2 preserva metadatos
            except Exception as e:
                print(f"  Error al copiar {nombre_archivo}: {str(e)}")
    
    print(f"\nTotal de categorías creadas/usadas: {len(categorias)}")
    print("Organización completada.")

def clasificar_archivo(nombre_archivo):
    """
    Clasifica un archivo según patrones específicos en su nombre.
    Retorna la categoría a la que pertenece.
    """
    # Comprobar patrones de flechas primero
    for flecha in ["Flecha Abajo", "Flecha Arriba", "Flecha Izquierda", "Flecha Derecha"]:
        if flecha in nombre_archivo:
            return flecha
    
    # Comprobar combinaciones de parte + ejecución
    for parte in ["RightArm", "LeftArm", "RightFist", "LeftFist", "RightFoot", "LeftFoot"]:
        for tipo in ["Moving", "Thinking"]:
            combinacion = f"{parte}{tipo}"
            if combinacion in nombre_archivo:
                return combinacion
    
    # Si ninguna combinación específica coincide, intentar con patrones más generales
    for parte in ["RightArm", "LeftArm", "RightFist", "LeftFist", "RightFoot", "LeftFoot"]:
        if parte in nombre_archivo:
            # Si tiene "Moving" o "Thinking" después
            if "Moving" in nombre_archivo:
                return f"{parte}Moving"
            elif "Thinking" in nombre_archivo:
                return f"{parte}Thinking"
            else:
                return parte
    
    # Si no se encuentra ninguna categoría válida
    print(f"⚠️ No se pudo clasificar: {nombre_archivo}")
    return None

if __name__ == "__main__":
    print("=================================================")
    print("  ORGANIZADOR DE DATOS EEG PARA EXPERIMENTOS")
    print("=================================================")
    main()