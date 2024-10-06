import pandas as pd

brands_url = ['subaru','mazda','mahindra','mini','dodge','citroen','mitsubishi','hyundai','ssangyong',
'mercedes-benz','nissan','bmw','chevrolet','toyota','jeep','dfm/dfsk','renault','audi','chery',
'zotye','seat','hafei','skoda','opel','changan','volkswagen','fiat','ram','suzuki','honda',
'great-wall','kia','jac','volvo','jaguar','peugeot','land-rover','daihatsu','byd','ford']

brands_homologue  = ['Dodge', 'Mazda', 'Mitsubishi', 'Nissan', 'Chevrolet', 'Renault', 'Hyundai','Mercedes-Benz', 'Ford', 
'Jeep', 'Toyota', 'Audi', 'Kia', 'Volkswagen', 'BMW','Peugeot', 'Honda', 'MINI', 'Chery', 'Suzuki', 'Land Rover', 
'BYD', 'Citroën','DFSK', 'Fiat', 'Volvo', 'Jaguar', 'Great Wall', 'Mini', 'Subaru', 'Ssangyong','Daihatsu', 'JAC', 
'Hafei','Skoda', 'Seat', 'Changan', 'Opel', 'Zotye']

bodywork_homologue  = ['NULO', 'Camioneta', 'Sedán', 'Hatchback', 'Furgón', 'Pick-Up', 'Coupé', 'Convertible','Van', 'Roadster']

transmision_homologue  = ['NULO', 'Automática', 'Mecánica']

fuel_homologue  = ['NULO', 'Gasolina', 'Gasolina y gas', 'Híbrido', 'Diésel', "Eléctrico"]

models_homologue = ['114i','118I','118i','120i','2','200','205','206','207','208','218','220i','250','3','300 SE',
'3008','301','306','307','308','320I','320i','325i','328i','330','330E','330I','350','370Z','405','407','420i','4Runner',
'5','500','5008','500X','508','550','6','626','700','A','A1','A200','A3','A35','A4','A45S','A5','A6','ASX','Accent','Accord',
'Active','Actyon','Alaskan','Almera','Altima','Alto','Amarok','Argo','Armada','Arona','Astra','Ateca','Atos','Aveo','B180',
'B2000','B2200','BT-50','Baleno','Baleno Cross','Beat','Beetle','Berlingo','Blazer','Bora', 'Bronco','Burbuja','C','C-Elysée',
'C180','C200','C3','C30','C300','C4','C5','C63', 'CJ', 'CR-V','CS 15','CS 35','CX-3','CX-30','CX-5','CX-7','CX-9','Caddy',
'Camaro','Captiva','Captur','Carens','Carnival','Cayenne','Celerio','Cerato','Cherokee', 'Chevy','Cheyenne','City','Civic',
'Clase A','Clase B','Clase C','Clase CLA','Clase E','Clase GL','Clase GLC','Clase GLE','Clase S','Clase SLC','Clase SLK','Clio',
'Clubman','Cobalt','Colorado','Commander','Compass','Cooper','Corolla','Corolla Cross','Corsa','Countryman', 'Creta','Cross Up',
'Crossfox','Crossland','Cruze','D-Max','D21','D22','DS3','DS7','Discovery', 'Drive','Durango','Duster','Duster Oroch','E-350',
'E-Pace','E200','Ecosport','Ecosport 2','Edge','Elantra','Eon','Epica','Equinox','Ertiga','Escape','Escarabajo','Esteem',
'Evoltis','Evoque','Expedition','Explorer','F-150','Fabia','Fiesta','Fiorino','Fit','Fluence','Focus','Forester','Fortuner','Fox',
'Freelander','Frontier','Fusion','Genesis','Getz','Gol','Golf','Grand Cherokee','Grand Move','Grand Pregio','Grand Vitara','Gravity',
'H1','H2','H3','HB20','HB20S','HB20X','HR-V','Haval','Hilux','Hybrid XV','I25','I30','I35','Ibiza','Idea','Impreza','Ioniq','Jetta',
'Jimny','John','John Cooper Works','Journey','Joy','Juke','K 2700','K05S','KO5S','KUV100','Kangoo','Kicks','Koleos','Kona','Korando',
'Kwid','Kyron','L200','L300','LC 70','LUV','Lancer','Leaf','Legacy','Leon','Logan','Luzun','M','M2','M235i','M240','M240I','M240i',
'M3','M340i','ML','ML350','MX-5','Macan','Magentis','Malibu','March','Megane','Mini Truck''Minivan','Minyi','Mobi','Montero', 
'Murano','Mustang','N200','N300','N400','Nativa','Navara','New Actyon', 'New Beetle','Niro','Nivus','Note','Octavia','Odyssey',
'Onix','Optima','Optra','Orlando','Outback','Outlander', 'Palio','Palisade','Panamera','Panamera S','Partner','Passat','Pathfinder',
'Patrol','Picanto', 'Pilot','Polo','Prado','Pregio','Primera','Pulse','Q2','Q3','Q5','Q7','Q8','QQ','QQ3','Qashqai','R18','R19',
'RAV4','RS3', 'Renegade','Rexton','Rio','Rodeo','Rodius','Rush','S-Cross','S-Presso','S2','S3','S4','S40','S60','SJ','SLC','SLK200',
'Sahara','Sail','Samurai','Sandero','Santa Fe','Santana','Saveiro','Scala','Scenic','Sedan','Seltos','Sentra','Sephia','Sequoia',
'Serie 1','Serie 2','Serie 3','Serie 4','Serie 5','Serie 6','Serie 7','Silverado','Sirion','Soluto','Sonet','Song Plus','Sonic',
'Sorento','Soul','Space Wagon','Spacefox','Spark','Spark GT','Sport Track','Sportage','Sprint','Sprinter','Stavic','Stepway',
'Stonic','Strada','Stylus','Sunny','Super carry','Swift','Symbol','T-Cross','T6','TT','Tahoe','Taos','Teramont','Tercel','Terios',
'Terracan','Tiggo','Tiggo 2','Tiguan', 'Tiida','Tivoli','Tonic','Touareg','Tracker','Trafic','Transporter','Traverse','Trooper',
'Tucson','Tundra','Tunland','Twingo','Uno','Urvan','V40','V5','Veloster','Vento','Veracruz','Verna','Versa','Virtus','Vitara',
'Vito','Vivant','Voyage', 'WR-V','WRX','Way','Wildtrak','Willy','Wingle','Wrangler','X-Trail', 'X-Type','X1','X2','X3','X4','X5',
'X6','XC40''XC60','XC60 T5','XC90', 'XE','XF','XJ','XL-7','XV','Xantia','Xpander','YOKI','YOYA','Yaris','Yaris Cross','Z20','Z4',
'Zafira','Zhongyi','Zotye','Zs',

# Pendiente de agregar en funcion de homologacion
'G63','HHR','530','Cupra Formentor','147','Polo Track','R4','Wagon R','Corvette','Paceman','316I','Grandland X','Grand Scénic',
'AMG','TI','Willys','Twizy','Santamo','CX-60','Defender','B2600','Velar','Mohave','YOYO','Eclipse','M135i','SX4','Tacoma','LC200',
'B1600','R9','Feroza','Parati','I18','318','Idolphin','LJ','Venue','Wingle 5','XC40','F-350','E-STAR','316i','960','318i',
'Gladiator','S7','Monza','i10','Qin','Zoe','Cx-90','Seal','F-250','FJ','K3','Q','Camry','Vito Tourer','F-Pace','F20','Yeti',
'X7','T3','Stilo','CX-50','Altea','Porter','Endeavor','220I','mx-30','Eclipse Cross','Duna','Yuan','Clase G','131','Cadenza','XC60',
'525i','CS15','Tribeca','GTX','C-10','Starex','EQE','121','Jumpy','IX','R8','Emotion','Fronx','Cabstar','180'


'F-100','Bora GLI', 'E-Tron Sportback','Glf','T8','Master','BX4Z','DS4','F-Type','Palio Adventure',
'BT50','Integra','Grand Carnival Sedona','Torres',
'TTS','Paseo','Jumper','iX3','320D','Nhr','128TI','Kombi', 'Ranger',
'520i','Carry','Milenio','M4','Quest','230','Montana'
]





cities = [
    ["cali", "valle-del-cauca"],
    ["rionegro", "antioquia"],
    ["itagui", "antioquia"],
    ["medellin", "antioquia"],
    ["bello", "antioquia"],
    ["sabaneta", "antioquia"],
    ["envigado", "antioquia"],
    ["la-estrella", "antioquia"],
    ["barranquilla", "atlantico"],
    ["teusaquillo", "bogota-dc"],
    ["fontibon", "bogota-dc"],
    ["suba", "bogota-dc"],
    ["puente-aranda", "bogota-dc"],
    ["martires", "bogota-dc"],
    ["la-candelaria", "bogota-dc"],
    ["usaquen", "bogota-dc"],
    ["santa-fe", "bogota-dc"],
    ["tunjuelito", "bogota-dc"],
    ["kennedy", "bogota-dc"],
    ["san-cristobal-sur", "bogota-dc"],
    ["barrios-unidos", "bogota-dc"],
    ["chapinero", "bogota-dc"],
    ["engativa", "bogota-dc"],
    ["antonio-narino", "bogota-dc"],
    ["rafael-uribe-uribe", "bogota-dc"],
    ["ciudad-bolivar", "bogota-dc"],
    ["duitama", "boyaca"],
    ["tunja", "boyaca"],
    ["valledupar", "cesar"],
    ["bogota", "cundinamarca"],
    ["mosquera", "cundinamarca"],
    ["fusagasuga", "cundinamarca"],
    ["cota", "cundinamarca"],
    ["funza", "cundinamarca"],
    ["cachipay", "cundinamarca"],
    ["girardot", "cundinamarca"],
    ["silvania", "cundinamarca"],
    ["zipaquira", "cundinamarca"],
    ["cajica", "cundinamarca"],
    ["chia", "cundinamarca"],
    ["neiva", "huila"],
    ["santa-marta", "magdalena"],
    ["villavicencio", "meta"],
    ["pasto", "narino"],
    ["cucuta", "norte-de-santander"],
    ["armenia", "quindio"],
    ["dosquebradas", "risaralda"],
    ["floridablanca", "santander"],
    ["bucaramanga", "santander"],
    ["ibague", "tolima"],
    ["buga", "valle-del-cauca"],
    ["jamundi", "valle-del-cauca"],
    ["cartagena-de-indias", "bolivar"],
    ["manizales", "caldas"],
    ["monteria", "cordoba"],
    ["pereira", "risaralda"]]
cities = pd.DataFrame(cities, columns=['ciudad', 'departamento'])