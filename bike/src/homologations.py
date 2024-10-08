import pandas as pd

brands_url = [
"agility", "auteco", "akt", "avanti", "apache", "ayco", "apollo", "artic-cat", "bajaj", 
"black-panther", "bmw", "benelli", "bera", "boss", "can-am",
"boxer", "bt", "bombardier", "cfmoto", "duke", "discovery", "ducati", "energy-motion", "harley-davidson",
"hero", "honda", "husqvama", "huqsvarna", "husaberg", "ktm", "kawasaki", "kinco", "kymco", "mrx",
"maverick", "monster", "polaris", "um", "united-motors"
"navi", "rs", "royal", "royal-enfield", "suzuki", "triumph", "tvs", "vento", "victory", 
"vespa", "yamaha", "ycf", "zontes"
]

brands_homologue  = [
'ADT','AKT','Agusta','Apollo', 'Aprilia','Arctic Cat','Army','Auteco','Ayco','BMW'
'Bajaj','Benelli','Bera', 'Black Panther', 'BT', 'Boss','Boxer','Can-Am','Cfmoto','Ducati','Energy Motion'
'Harley-Davidson','Hero','Honda','Huqsvarna','Husaberg','Kawasaki','Kymco', 'MRX', 'Maverick','Monster',
'Polaris','Pulsar','Royal Enfield','Suzuki', 'RS', 'TVS','Triumph','UM','United Motors','Victory', 'Vento',
'Vespa', 'YCF','Yamaha','Zontes'
]

moto_types = [
'Naked', 'Street', 'Scooter', 'Touring', 'Deportiva', 'Enduro',
'Doble propósito', 'Cuatrimoto', 'Chopper', 'Custom', 'Motocross',
'Motocarro', 'Crucero', 'Mini moto', 'Scrambler', 'Adventure',
'Eléctrica', 'Hypermotard', 'Turismo', 'Shopper', 'Clásica',
'Supermoto', 'Cafe racer', 'Retro vintage', 'Superbike']

models = {
# 1 palabra
"1w": [
'mt07', 'mt01', 'mt09', 'mt03', 'gsxr150', 'f900xr', 
'hypermotard', 'mrx', 'dyna', 'fz25','psx', 'ttr50', 
'ax4', 'icon', 'hunk','r1250gs', 'n-max', 'dio', 
'adventure', 'xr650l','fzn250-a', 'x-town', 'ds','adventour',
'402s', 'bomber', 'panigale', 'pcx160', '390', 'er6f',
'450mt', 'tiger', 'versys', 'eco', 'varadero', 
'scrambler', 'xblade', 'dominar', 'gixxer', 'cb190r', 'g310r',
'gsx-r150', 'fazer', '162', 'gsx-s150', 'nc750x', 'szr',
'twist', 'ns160', 'dynamic', 'avenger', 'discover', 'monster',
'aerox', 'd250', 'f900r', 'ycz', 'sportster', '250sr',
'supersport', 'sp', 'evo', 'fzs', 'fz-150', 't3', 'r15',
'xmax', 'nkd', 'cr4', 'bolt', 'tracer', 'mt15', '650mt',
'streetfighter', 'xt660', 'z650','vxl', 'sz', 'avenis',
'kle650', '125-2a', 'bws', 'modelo', '890', 'ycf', 'xr150r',
'supermeteor', 'sv650/a', '100', 'himalayan',
'hunter', 'xr', 'klx150j', 'xtz', 'black', 'outlander',
'gsx-r600', 'utv300x', 'fz', 'x-max', 'tnt25',
'180s', 'commader', 't2', 'cb160', 'ranchera', '125', 'maxi', 'dt',
'boneville', 'ac300', 'v-rod', 'carguero','buggy',
'fusión', 'ct100es', 'gsx-8s', 'moped', 'dash',
'n250', 'intruder', 'raptor',
'ntorq', 'bwsx125', 'renegade', '550', 'one', 'z900', 'srcambler',
'twister', 'crf110f', 'boss', 'tnt150i', 'vws', 'plr70',
'interceptor', 'sr250', 'plr450', 'sport','bs200',
'gsx125', 'gsx-s1000', 'acces', 'rsv4',
'f800', 'cb1000r', 'vfr', 'duke', 'ee5', '750', 'r6s',
'scram', 'xmax300', 'utv', 'r1250r', 'c150', 'tt', 'dcx88c',
'752s', 'xre190', 'f650st', 'u1-l3', '150x', 'hayabusa',
'adv750', 'rx2', 'plr90', 'gsx-r1000r', 'plr250',
'multistrada', 'primavera', 'apache', 'magna', 'r15v3', 'rx300-4',
'ts', 'v-strom', 'gz', 'r3', 'x-pro', 'tt125', '300', 'x-adv750',
'reactor', 'xlr', 'venom', 'ranger', '100t', '103',  'n160',
'dominar', 'crypton', 'rx', 'forty-eight',
'xpulse', 'fzs6vr', 'fz16', 'ttr', 'transalp', 'superlow', 'fz250',
'calibmatic', 'raider', 'cb650r', 'bold', 'life', 'xtz150',
'switch', 'fusion', 'crf450r', 'r18rre99am', 'special', 'x-adv',
'200', 'aiii150', 'fz8s', 'cr125', 'rx-115', 'z250sl',
'r6r', '660r', 'xrc', 'enduro', 'apollo', 'combat',
'adv750m', 'gs750', 'rs200', 'sz15rr', 'jer', 'aiii125', 'nk250',
'flex', '180fi', 'caponord', 'fzn250', 'libero', 'ninja', 'cb300f',
'intrepid', 'c125', '350w', 'rx1', 'rx-100', 'lite',
'zh2', 'cbf150', 'ceronte', 'mo', '302s', 'cr4-162', 
'1100nv', '87', 'ltz400', 'comm200', 'agility', '400', 'pcx',
'svartpilen', 'fzn150d', 'splendor', 'address', 'fz1', '310r',
'vstrom', 'xtz660', 'nc750xdp', 'xre300abs', '821',
'xpulsefi', 'ktm890r', 'c70', 'x-blade', 'le650f', 't-max',
'z1000b', 'crf150r', 'trident', '950sp', 'ak125rl',
'z400', '155u', 'r-nine', 'p150', 'ignitor', 'ninet', 'chr',
'torito', 'cb175', 'hymalayan', '1000x-4', 'breakout', 'crf50f',
'gsx-r750', 'click', 'dr200', '200r', '250', 'bomber150', 'zx6r',
'multiestrada', 'dr50', '350-9', 'ay200zh', '3w', 'gs650',
'zr1000', 'xpedition', '250n', '790', 'arizona', 'r7', 
'bwsx', 'adv750k', 'u1', 'dominar250', 'zx10r',
'cbr1000rr', 'freewwing', 'crf250r', 'hypermotar', 'fxsts',
'tvs', 'vt750dc', 'kx60', 'ttds', 'gs310', '1200z',
'ex300', '800', 'xadv', 'nmax', 'n-max',
'avanti', 'e3', 'bengala', 'f40', 'cbr250', 'diavel', 
'starker', 'shock', 'dior', 'avance'],

# 2 palabras
"2w": [
'tenere 660', 'pioneer 1000-3', 'r1300 gs', 'crf 150r', 'wr 250f',
'690 smcr', 'c 70', '160 r', 'zontes 155', 'active 1',
'450 exc-f', 'me 17b', 'rm-z 450', 'gs 1200',
'nht 190', 'vulcan s', 'cb160f std', '650 mt',
'yz 65', 'xdiavel s', 'odyssey 350', 'vrsc awa', 'talon 100r',
'yfm 700r', 'advance r',
'supertenere 1200', 'gk 350', 'cb650 r',
'klz1000 acf', 'crf 1100', 'vulcan 650s',
'dinamic pro', '850gs trophy',
'dr 150', 'dominar 400', 'cb125f', 'rallye ss', 'ns 150',
'xtz 250', 'transalp 650', '890 gp', 'gixxer 150', 
'versys 650', 'pulsar 125', 'g 310', 'mt 03',
'ultra limited', 'agility 150', 'adventure 250', 'agility 125',
'dominar 250', 'africa twin', 'grizzly 700', 'boxer 100',
'ttr 200', 'duke 790', 'dt 125', 'multistrada 1200', 'mt 15',
'classic 350', 'mt 09', 'bws fi', 'fly 125', 'rockz 125',
'xsr 900', 'downtown 300i', 'svartpilen 401', 'xre 300',
'cb160f dlx', 'fzn 250', 'yz 450f', 'meteor 350',
'fazer 150', 'gixxer 155', 'leoncino 250', 
'svarpilen 200', 'venon 250', 'pulsar 180', '701 supermoto',
'te 300', 'rtr 200', 'boxer 150', 'gs 500', 'dynamic pro',
'1290 r', 'chr 125', 'r ninet', 'fz 250', '390 duke', 'shark 200x',
'agility rs', 'spider 200', 'klr 650', 'black 171', 'shark 200',
'norden 901', 'fe 350', 'wave 110', '125 jet', 'fz6 fazer',
'apache rtr', 'ninja 300', 'desert x', '890 adventure',
'tnt 150i', 'hunter 150', 'crf 250', 'willcat 700',
'er 6n', 'victory black', 'hypermotard 939', 'cbr 250r',
'fino af115s', 'xtz 125', 'apache 160', 'duke 200', 'pulsar 150',
'rzr 800', 'yzf r15', 'gsx 125r', 'nkd 125', 
'xt 225', 'trk 502', 'grizzly 450', 'n-max 155', 'kawasaki 300',
'er6f 650', 'interceptor 650', 'dt 175', 'z 1000', 'gixxer 250',
'duke 250', 'xtz 250z', 'duke 390', 'mawi 125', 'z 900',
'life 125', 'outlander 1000', 'sxf 250', 'gs 310', 
'z 250', 'nqi gts', 'xtz 150', 'cb 650r', 'xt 660', 'nmax s',
'tnt 150', 'dt 230', 'f900 r', 'cb 125', 'svartpilen 200',
'701 enduro', 'rumbler 350', 'an 125', 'nmax 150i', '350 fe',
'pcx 150', 'f900 xr', 'gx 250', 'tiger 800', 'gsx-s 750',
'gn 125f', 'raptor 125', 'crf250 rx', 'rtx 150', 'tnt 250',
'raptor 250', '790 duke', 'multistrada 1200s', 'bomber 125',
'monster s2r800', 'svartepilen 200', 'outlander max', 'xr 150',
'rc 200', 'yzf r6', 'cforce 550', 'bold 125', 'duke 890r',
'super meteor', '700 mt', 'nmax 155',
'v-strom touring', 'ninet scrambler', 'scram 411', 'sun bike',
'avenger 220', 'gn 125', '350 t1', 'maverick 1000r',
'vitpilen 401', 'raptor 90', '390 adventure', 'ct 125',
'duke 890', '180 s',
'boxer x150', 'multistrada v2s',
'dl 650', 'advance 110', 'v-strom 650xt', 'multistrada 1260',
'cb 1000r', 'r 1200', 'bwis fi', 'new monster', 'avanty x',
'vtx 600', 'fe 250', 'inazuma 250', 'hunk 190', 'pulsar n250',
'pulsar 135', 'cb 110', 'eco 100', 'gsx 150', 'dt 200', 'xr 250',
'rr 310', 'bruterforce 300', 'pulsar 180gt', 'ern 650', '125 f1',
'monster 797', '350 r1', 'dr 650', 
'nmax connected', 'versys 1000', 'adventure 390', 'rzr xp',
'gs 650', 'vstrom 1000', 'black shadow', 'iron 883', 'cr5 180',
'buggy xr', 'discovery st-r', 'scram himalayan', '250 duke',
'versys 300', 'gz 250', '200 ng', '250x pro', 'ninja zx-6r',
'ape 501', 'cbx 1047', 'cbf 125', 'r1200 gsa', 'dyna swtichback',
'venom 250', 'v-strom 250', 'twist 125', 'mt 10',
'bandit 650', 'c400 gt', 'maverick x3', 'g650gs dakar', 'gsx s',
'trk 502x', 'cb 190', 'cb 1000', 'monster 796', 'xmax 300',
'shadow 750', 'flex 125', 'cr4 125', 'bomber 150', 'ttr 125',
'street triple', 'ttds 200', 'boxer s', 'cr4 162', 'tenere 700',
'k25 adventure', 'forty eigh', 'rs 200', 'dash 110', 'fe 450',
'pcx 160', 'raider 125', 'bws f1', 'pro xp', 'rev 180', '300 sr',
'bwsx 125', '300 rally', 'trk 251','cb 160',
'fat boy', 'victory 125', 'gts tft',
'hunk 160r', 'gsx- 8s', 'boss js', 'tiger sport',
'xj 750', 'ntorque 125', 'commander 1000', 'crf 80f', 'agility go',
'yzf r3', 'pulsar speed',  'xtown 300', 'lt 80',
'boxer 150x', 'ybr 125', 'ignitor 125', 'tc 65', 'mavericks xrc',
'bws x', 'rs 660', 'crox 125', 'diavel v4', 'magna 750',
'sportster low', 'ce 04', 'aiii 110', 'starker hunter',
'rumbler 500', 'fc 250', 'st 1200z', 'klx 150', '890 duke',
'501 fe', 'switch 150', 'mrx 125', 'sportster 883',
'tiger 1200', 'nc750 xd', 'monster 1200', 'zontes 310m',
'cr4 162cc', 'tuareg 660', 'f700gs premium', 
'crf 150rb', 'tdm 900', 'venom 14', 'hunter 350', 'twister 250',
'cb 125f', '1090 adventure', 'access 125', 'klx 250', 'super 8',
'thriller pro', 'continental gt', 'gsr 600', '310 r', 'r1250 gs',
'transalp 600', 'szr 150', 'pulsar p150', '350thunderbird x',
'dynamic k', 'vitpilen 701', 'ttr 180', 'bomber 250',
'2023 connect', 'xpulse 200', 'v-strom 800de', 'special 110',
'adventure 890', 'modelo 2012', '250 adventure', 'mt 07',
'domina d250', 'hypermotard 950', 'platino 100', 'monster 821',
'road star', 'gixxer 154', 'street twin',
'dominar ug', 'v 80', '350 t2', 'vstrom 250sx', 'panigale v4',
'dl 1000', 'gz 150', 'classic 500', 'nk 250',
'scrambler 900', 'mrx arizona', 'pulsar n160', 'sv 650',
'leoncino 500', 'one mp', 'adventure 790', 'agility 2015',
'mrx 150', 'pulsar 200', 'kymco twist', 'at 650', 'evo 150',
'falcon 125', 'z 800', 'ysf r6', 'xadv 750',
'vstrom xt', 'qr 50cc', 'rocket 125', 'burgman 125', 'nitro 125',
'vxl 150', 'c600 sport', 'bws 125', 'sx-f 350', 'sbr vinotinto',
'fxdrs 114', 'yw 125', 'gixer 250', 'hunk 200r', 'buggy 200xr',
'nmax conected', 'jet 125', 'rtr 180', 'vulcan 900', 'tuono 660',
'r1250gs adventure', 'scrambler icon', 'ns 200', 
'xr 190', 'maverick rs', 'commander 125', 'sf fi', 'ntorq 125',
'cbr 500', 'fz 160', 'fz 150', 'xblade  162', 'outlander xt',
'outlander 650', 'monster sp', 'sz r16', 
'vstrom 650', 'nmax connect', 'cbf 150', 'xr 190l', 'exc f',
'gs 750', 'v-strom 650', 'monster 937', 'jet 14', 'jet 150',
'renegade commando', 'icon dark', 'v-strom 650dl',
'350 exc-f', 's1000r  m', 'versys 1000s', 'cbr 1000rr', 'bee 350',
'tb-x 350', 'rtr 160', 'victory one', 'iron 1200',
'x-adv 750', 'heritage softail', '250 xp', 'agility fusion',
'crf 250f', 'outlander 800', 'discover 125', 'himalayan 411',
'rally edition', 'panamerican special', 'address nm', 'as 150',
'fazer 250', '250 nk', 'crf 250rx', 'z250 sl', 'cr4 200',
'yzf r1', 'x-max 300', 'eco deluxe',
'elite 125', 'f 900r', 'r1200 adventure', 'start 125',
'x3 xrc', 'bonneville t100', 'dr 200s',
'suzuki ltz-400', 'super moto', 'can am',
'fat bob', 'f900 gs', 'yz 250', '790 r', 'best 125', 'xre 190',
'dinamic 125', 'ntorq tvs', 'z900 naked', 'z 50', 'apache 200',
'bonneville t120', 'gts 300', 'combat 125', 'sport 100', 'fz 50',
'cbr 929rr', 'ninja 400', 'dash 125', 'super bike',
'special 100', 'cripton fi', 'r1200gs 2017', 'atv 110r',
'wave 100', 'z 400', 'go 125', 'agility fly', 'kmx 125',
'1290 super', 'gs 800f', 'tenere 250', 'g310 r',
'650 nk', 'ttx 200', 'sz r150', 'wawe 110', 'v-strom 650a',
'xblade 160', 'ninja 250', 'cbf 160',
'gts super', 'dominar 250r', 'trx 500fe', 'boxer 125',
'downtown 300', 'rx 200', 's 1000', 'yz 125', 'adventur  250',
'xt 500', 'bws 2', 'cbf max', 'n-max connected', 'cr5 200',
'vfr 750', 'carguero 200', 'r1 350', 'x-advr 750', 'rayker rally',
'xtown 300i', 'super soco', 'adventure 790r', 'trazer 190i',
'fazer 1000', 'ycz 110', 'softail breakout', 'duke 690', 'mrx pro',
'ttx 180', '502 c', 'trident 660', 'rc 390',
'shock 115', 'v-strom 1050de', '125cc raptor', 'mt 01',
'streetfighter v4s', 'vstar 650', 'custom 1670', 'cbr 1000rrc',
'duke r', 'ninja zx-10r', 'adventure 1190', 'crypton fi',
'tracer gt', 'tc max', 'nd 300', 'multistrada 950', 'gsx 125',
'400 ug', 'ts 125', 'scrambler 1100', 'stryker 125', 'exc 250',
'gixxer fi', 'step 125', 'trail s', 'trx 250', 'lets 110',
'benelli 180s', 'zontes 155u', 'jet14 150', 'sz rr',
'ranger 570sp', 'breakout 114', 'cbr 250', 'yz 85', 'hunter 160',
's adventure', 'vfr 1200', 'nk 400', 'crypton 115', 'libero 125',
'xe 125', 'advenger 220', 'le300 versys',
'raptor 700', 'n-max coneccte', 'burgman fi', 'pulsar 220',
'pulsar 180ug', 'v strom', 'xadv x-adv', 'sr 160', '990 duke',
'fazer 16', 'vrsc v-rod', 'vlx 600', 'x blade',
'wave 110i', 'crf 110f', 'panigale 899', 'splendor 110',
'adventour 250', 'susuki 650', 'fiddle iii', 'rzr 1000',
'karizma zmr', 'x adv', 'ybrz 125',
'xciting 400', 'atv 250f', 'thriller 200', 'nqi sport', 'nhx 190i',
'neo nx', 'gs 1250', '1200 custom', '50 sx',
'urban s', 'f750gs premium', 'fz 1', 'thriller 150',
'310 rr', 'x max', 'ax4 115', 'lets 112', 'gsf 650',
'turismo veloce', '250 racer', 'supersport s', 'sr 250',
'xr 150l', 'versys 650', 'nkd 125','street glide', 'super adventure',
'mt 09', 'v-strom 650', 'defender max' 'ns 125',  '180 fi',
'multistrada v4s',  'evo r3', 'ns 150', 'ns 160', 'ra1250', 
'n 250', 'n 160', 'multistrada 1200', 'sportest xl',  'jet 14', 
'duke 390', 'avanti', 'new black', 'ninja 400','bn 302',  
'bmw f900 r', '200 ns', 'tdm 900','fz 25', 'versys x', 'pcx 160',  
'gsx s150','v-strom 1000', 'duke 690','bn 251','dl 650', 'panigale 899', 
'tiger 1050', 'gixxer fi', 'n-max 155', 'as 200',  'rs 200','speed 135',
'sportsman 450','boxer 150x','apache 200', 'agility go', 
'gixxer 250', ],

# 3 palabras
"3w": [
'g 310 r', 'xpulse 200 fi', 'maverick x3 xrs',
'r 1250 gs',  
'bws at 125', 'viva r style',
'super sport 950', 'apache rtr 200', 'gixxer sf 250', 'c 400 x',
'defender max hd7', 'f 800 gs', 'f 850 gs', 'g 650 gs', 'r 1200 r',
'gixxer 250 fi', 'brute force 750', 'tracer 900 gt',
'gixxer sf 155', 'f 900 xr', 'versys x 300', 'apache rtr 160',
'r 1200 gs', 'cb 160 f', 'nc 750 xdm', 'f 700 gs',
'zontes 350 t2',  'tt ds 200',
'gixxer fi 155', 
'duke 200 ng', 'f 800 r', 'xtz 125 e', '1090 adventure r',
'gixxer sf fi', 'mule pro fxt', 
'scrambler desert sled', 'g 310 gs', 's 1000 rr', 'f 750 gs',
'1250 gs adventure', 'cb 500 x', 'r 1300 gs', 'jet 5 150',
'tiger rally pro', 's 1000 xr', 'crf 250 f', 'sport 100 els',
'hunk 190 r', 'rninet urban gs', 'street triple r',
'streetfighter v4 s', 'xf 650 freewind', 'dynamic r 125',
'bws x motard',
'dio 110 dlx', 'bws 125 fi', 
'outlander x mr', 'cb 160 dlx', 'tt 250 adventour',
'apache rtr 180', 'r1250 gs rallye', 'super adventure s',
'xre 300 abs', 'cbr 1000 rr', 'goldwing gl 1200', 'virago xv 750',
'bws 100 at', 'hunk 190 fi', 
'fino cc 115', 'sz rr 150', 'xt 660 r', 'bmw f800 gs',
'pcx 150 dlx', 'gixxer 250 sf', 's 1000 r',
'ducati scrambler 1100', 'wave 110 s', 
'125 dash 2023', 'victory life 125', 'dio dlx 110',
'f850 gs premium', 'xpulse 200 advance', 'smc r 690',
'hypermotard 950 sp', 'xt1200ze super tenere',
'trx 520 foreman', 'f 900 r', 'c 650 gt', 'cb 190 repsol',
'dynamic pro 125', 'dr 150 fi', 'r 1250 r',
'multistrada 1260 s', '790 adventure s',
'r 15 v3', 'f 650gs twin', 'magna 750 tour', 'ninja ex 650r',
'sef 300 factory', 'v-rod night special', 'duke 390 ng',
'vstrom dl 1000', 'agility city 150', 'boxer ct 125', 'x max 300',
'bullet classic 500',
'gs 650 sertao', 'hunk 160 rs', '890 duke gp', 'z 250 sl',
'dyna wide glide',
'z 900 rs',
'c 650 sport', 'discovery 125 str', 'multistrada 1200 stouring',
'ninja ex 250', 'dominar 400 ug', 
'dinamic pro 125', 'apache 180 2017', 
'street triple rs', '300 exc tpi', 'x pulse 200',
'freeride 250 2t', 'tiger 800 xrx', 'rallye ss hp', 
'raider 125 racing', 'boxer ct 100', 'duke 690 r',
'hypermotard rve 950', 'r1250gs adventure trophy',
'rincon trx 650', 'ninja ex 300', 'bws x 125', 'rs 125 my2017',
'xvs 950 a', 'r 18xa0 classic', 
'nmax 155 connect', 'norden 901 exp',
'benelli tnt 150', 'pioneer limited 1000', 'cb 650 r',
'maverick x3 rs', '790 adventure r', 'tuono v4 1100',
'cb 250 twister', '990 superduke r', 'xpulse kit rally',
'xj6 diversion n', 'cb 125f max', 'fz 25 2024',
'v-strom 250 sx', 'klx 300 r', 'yamaha xj6n 2013',
'super adventure 1290', 'xtz 150 arena',
'apache 180 2v', 'fz 150 2.0', 'apache 160 4v',
'nmax connected 2022', 'tracer 9 gt', 'freeride 2t 250',
'wr 450 f', 'gs 1200 adventure',
'multistrada 1200 v4s', 'super meteor 650', 'crypton t115 fi',
'dio led 110', '200 ng duke', 'ns 200 fi',
'gn 125 nova', 'v-strom dl 650', 'xt1200z super tenere',
'dominar 250 ug', 'cm 200 t',
'adventure 1190 r', 'r1250 triple black', 'tt dual sport',
'adress ag 100', 
'fzn 150d-6 (fz-s)', 'scrambler 1200 xc',
'xt 1200 ze', 'mini bike street',
'xr 250 tornado', 'nqi sport gts', 'f850gs premium hp',
'multistrada v4 rally', 'nmax 155 connected', 'duke 250 ng',
'boxer ks 100', 'agility naked 125',
'duke 890 r', 'ninja 400 krt', '250 duke ng',
'sportster xl 883n', 'ttr 200 ns', '390 duke ng',
'multistrada v4s rally', 'multistrada pikes peak', 'shark x 200',
'boxer ct 100es', 'honda tornado 250', 'farmer pro x', 'sr gt 200',
'tiger explorer 1200', 'ax 4 e3', 'cr4 162 cc',
'n-max 155 connected', 'vstrom dl dirt', 'r 18 transcontinental',
'viva r cool', 'v strom 1050', 'maverick x3 xmr',
'sportster s 1250', 'virago xv 1100',
'ktm sx-f 250cc', 'vstrom 250 dl', 'trk 502 x', 'discover 150 st']

# Revisar
# 'ttr 125 2023', 'sportster iron 883',
# 'dominar 400 touring', 'victory one 100', '530 enduro rally',
# 'multistrada v2 s', '950cc hypermotard rve', 'flhtk ultra limited',
# 'f750 gs premium', 'ak chr 125', '690 enduro r',
# 'brutale 800 dragster', 'f 650 gs', 'yw 125 xfi',
# 'victory switch 150', 'v strom 650', 'ns 160 fi', 
# 'xpulse 200 t', 'xl883n kit 1250', 'super tenere 1200ze',
# 'nkd 125 metal', 'r 1200 rt', 'thruxton 1200 rs', 'k 1300 r',
# 'cb125f max 2023', 'xmen v1 pro', '390 duke g3',
# 'leoncino 500 trail', 'dl 650 xt', 'super glide 2007',
# 'rzr (2 puestos)', '701 enduro lr', '990 adventure r',
# 'nmax connected 155', 'agility go 125', 'zt 310 m',
# 'f700 gs premium', 'xadv 750 cc', 'rzr turbo r', '150 fi 2023',
# 'street scrambler 900', 'ayco250. doble pacha', 'dr 200 se',
# 'gts 300 supertech', '350 six days', 'multistrada 950 s',
# 'nmax connet 155', '890 adventure r',
# 'versys 300 adv', 'k 50 gs1200', 'gs 1250 adventure',
# 'ryker rally edition', 'ltf-250 modelo 2004', 'zontes 350 r1',
# 'xciting r 300', 'agility 125 2.0', 'r1250 gs adventure',
# 'bwis x fi', 'tiger explorer 1.215cc', 'r 1150 r',
# 'yamaha ybr 125', 'pitbike 50 cc', 'versys 1000 r',
# 'shadow vt 600', 'carpado re4s petrol', 'xre 300 adventure',
# 'cb 1000 r', 'angility alli new', 'super adventure r',
# 'ktm rc 200', '1250 gsa adventure', 'norden 901 adv',
# 'neo max 100', 'xtz 250 lander', 'pulsar 135 speed',
# 'side by side', 'n max 155', 'thriller 150 i3s', 'sym crox 150',
# 'vulcan 900 classic', 'vstrom 650 dl', 'nkd led 125',
# 'discover 100 sport', 'dynamic pro cbs', 'boxer 150 x',
# '250 duke g3', 'vf 750 c', 'victory zs 125', 'n-max connect at155',
# '155 fi gixxer', 'bmw gs 1200', 'starker skuty sport',
# 'mt 09 tracker', 'venom 250 pro', '450 exc racing', 'vc 200 naked',
# 'fz n 250', 'king quad 500', 'r 1150 gs', 'scrambler 1200 xe',
# 'rm z250 e', 'invicta cb 150', 'discover 125 str',
# 'n max connected', 'fat bob 114', 'versys 1000 se',
# 'multistrada 1260 enduro', '180 fi neón', 'cb 500 f', 'xt 1200 z',
# 'cr4 125 cc', 'bet & win', 'dyna street bob', '1090 adventure s',
# 'seventy - two', 'rr 4t 350', 'dynamic klassic 125',
# 'pulsar 180 neon', 'dinamic r 125', 'duke 200 wo', '180 s prox',
# 'navi honda 2022', 'commander max xt', '250 exc tpi',
# 'nkd 125 digital', 'jet 4 125', 'gs 1200 r', 'jet 14 xc15w',
# 'shadow vt 1100', 'multistrada v4 s', 'mt 09 tracer',
# 'street twin ec1', 'rzr (4 puestos)', 'xciting s 400',
# 'tiger 800 xr', 'bombardier ds 650', 'r1200 gs k50', 'gsx - 8s',
# 'dyna superglide cust', 'mt 09 sp', 'moto carro piaggio',
# 'pulsar ns200 ug', 'crf 250 rx', 'yzf r15 v3', '1190 adventure r',
# 'xr 400 r', 'maverick x3 max', 'voge 300 ds', 'pulsar 135 ls',
# 'gixxer 155 sf', 'cb 190 racing', 'f 900 gs', 'eco deluxe 135',
# 'dream neo 110', 'road glide special', 'pulsar r 220',
# 'hunk sports 160r', 'evo r3 125', 'talon x 4',
# 'ultramatic grizzly 660', 'zontes t2 350', 'wr 250f 2013',
# 'dr150 modelo 2024', 'ktm adventure 250', 'outlander limited 1000',
# 'xv 250 virago', 'r 125o gs', 'multistrada v2s 950',
# 'cb 190r repsol', 'maverick ds max', 'cb 125 e', 'agility all new',
# 'tt 200 ds', 'pulsar p 150', 'new life 125tk',
# 'fz 2.0 /150', 'hypermotard 821 sp', 'dynamic 125 sc',
# 'bmw gs 800', 'x blade 2023', 'v-strom 1000 xt',
# 'multiestrada v4 s', 'xt 660 x', 'hypermotard 698 mono',
# 'ignitor ss 125', 'thriller sports 150',
# 'bajaj dominar 400', 'versys 1000 s', 'tj006 tiger 110-650',
# 'xt 660 tenere', 'eco deluxe i3s', 'tiger 900 rally', 'nc 750 dtc',
# 'heritaje motor 114', 'x blade 160',
# 'tiger sport 660']

# # 4 palabras
,
"4w": [],
# 'street fighter v4 s', 'pulsar ns 200 fi', 'r 1200 gs adventure',
# 'gixxer 150 sf fi', 'venom 18 (200 cc)', '1290 super adventure s',
# 'fz 250 edición especial', 'pulsar ns 160 fi', 'r ninet urban gs',
# 'tvs apache rtr 160', 'dr 150 fi abs', 'gixxer 150 fi abs',
# 'xpulse 200 fi abs', 'adv 390 adventure 390',
# 'r 1250 gs adventure', 'f750 gs premium lk', 'f 850 gs adventure',
# 'jet 5 r 150', 'outlander max xt 850', 'fz 25 edición especial',
# 'r1250gs triple black hp', 'ns fi 200 abs', 'super moto 690 r',
# 'victory one st 100', 'f 800 gs adventure', 'apache rtr 200 4v',
# 'ns 160 fi 2022', 'xl883n subida a 1200', 'bmw r 1250 gs',
# 'bóxer ct 100 ks', 'gixxer sf 150 abs', 'boxer ct 100 es',
# 'apache rtr 160 4v', 'ktm 790 adventure r', 'duke 250 gen 3',
# 'pulsar ns200 fi abs', 'dominar 250 fi abs', 'pulsar 180 fi neon',
# 'cr4 200 cc pro', '1290 super adventure t', 'defender max hd8 dps',
# '50 cc 2 t', 'royal enfield scram 411', 'shadow 750 mod 2007',
# 'africa twin crf 1000dh', 'apache 200 fi abs',
# 'outlander max dps 500', 'apache 310 rr racing',
# 'ducati scrambler tribute pro', 'apextra 501 pick up',
# 'apache trt 180 2v', 'multistrada v4 pikes peak',
# 'hunk 160 stealth rs', 'tiger rally pro 900', 'apache rtr 200 fi',
# 'viva r 115 style', 's 1000 rr hp4', 'pulsar 200 ns pulsarmania',
# 'gixxer fi abs 150', 'x max 300 2020', 'gn 125 euro 3',
# '250 sx – f', 'suzuki vstrom 250 sx', 'x max 300 xmax',
# 'bmw r nine t', 'maverick sport max dps', 'street triple rs 765',
# 'ttds 200 edition rally', 'nkd 125 slipper clutch',
# 'fz 16 st fazer', 'g 310 gs rally', 'maverick x3 max xrs',
# 'mt 09 tracer gt', '1290 super adventure r',
# 'interceptor 650 baker express', 'agility 125 dgtl 3.0',
# '1290 super duke gt', 'wildcat 1.0 turbo 4x4',
# 'discover 125 st bajaj', 'gs 1200 versión rally',
# 'africa twin crf 1.100', 'super soco tc 1900',
# 'classic 350 abs fi', 'ultra classic tour glide',
# 'crf 1000 afrika twin', 'r nine t scrambler',
# 'royal enfield himalayan 411', 'jet 5r 150 cc',
# 'apache rtr 160 2v', 'pulsar 180 fi 2025', 'sr 160 mt abs',
# 'ax4 evolution euro 3', 'pit bike 90 cc', 'x3 xrs smart shox',
# 'rzr 14 1000 xp', 'gs 1250 hp ss', '1200 gs k 50',
# 'tvs ntorq 125 2022', 'thriller pr 150 i3s', 'rzr 1000 4 ptos',
# 'apache rtr 160 v2', 'boxer ct 100 ks', 'g 650 gs sertao',
# 'victory switch 150 2022', 'scrambler sport pro 1.100',
# 'dynamic pro cbs 125', 'bomber 150 es especial',
# 'electra glide ultra limited', 'rzr turbo r 4', 'ranger rzr 4 800',
# 'r 1200 gs k50', 'r 1200 gs ad', 'ape xtra x chasis',
# 'ktm 390 duke g3', 'bmw gs 800 adventure', 'apache rtr 200 abs',
# 'citycom 300 i cbs', 'super duke r 1290', 'v strom 250 sl',
# 'pulsar 180 neón fi', 'gsx 650 fa abs', 'r1300 gs triple black',
# 'agility kymco twist 125', '525 xc desert racing',
# 'ktm 200 duke wo', 'tuono v4 1100 factory',
# 'maverick xrs smart shox', 'cb 125 fmax ii', '200 duke ng silver',
# 'gixxer fi sin abs', 'tiger 900 rally pro', 'pcx 150 - dlx',
# 'defender max dps hd9', 'xt 660 z ténéré',
# 'multistrada v4 pikes peake', 'ntorq fi xconnect 125',
# 'exc six days 450', 'gixxer sf fi abs',
# 'scrambler 803 desert sled', '690 enduro r abs',
# 'pulsar ns 200 ug', 'r 1250 gs odc03e', 'rzr 1000 trail s',
# '300 exc sixdays tpi', 'tt ds 200 rally', 'africa twin 1.100 d4',
# 'xr650 l 650 mt', '350 exc – f'
}
models_all = models["1w"] + models["2w"] + models["3w"] + models["4w"]

abreviaturas = [
'mt', 'dt', 'fz', 'cb', 'dr', 'gn', 'xr', 'yz', 'f1', 'er',
'6n', 'rc', 'fe', 'gs', 'rs', 'r6', 'r3', 'go', 'fi', 'tc', 'xt',
'dl', 'v4', 'gt', 'js', 'gx', 'xp', 'nk', 'te', 'gp', 'gz', 'x3',
'at', 'sv', 'sz', 'rr', 'yw', '8s', 'r1', 'fc', 'sr', 'xj', 'ts',
'ct', 't1', 'ug', 'an', 'nm', 'mp', 'ns', 'sf', 'xd', 'rx', 'ce',
'as', 'e3', 'qr', 'sp', 'ng', 'nd', 'lt', 't2', 'xe', 'am', 'sl',
'sx', 'nx', 'st', "ex", 

'xtz', 'xre', 'yzf', 'nkd', 'ttr', 'trk', 'pcx',
'bws', 'cr4', 'tnt', 'gsx', 'r15', 'cbr', 'rtr', 'rzr', 'crf',
'klx', 'cbf', 'klr', 'ybr', 'fz6', 'cr5', 'dlx', 'mrx', 'rtx',
'gsr', 'ape', 'ttx', 'nqi', 'exc', 'gts', 'xsr', 'v2s', 'vxl',
'ern', 'sxf', 'xrc', 'sbr', 'fzn', 'r16', 'tft', 'tdm', 'kmx',
'gsa', 'v4s', 'bee', 'trx', 'ysf', 'chr', 'cbx', 'vtx', 'szr',
'k25', '796', 'iii', 'atv', 'vfr', 'f40', 'rev', 'adv', 'vlx',
'bs6', 'ycz', 'ycf', 'zmr', 'nhx', 
'r6r', 'vws', 'zh2', 'r6s', 'gsf', 'psx', 'fz1', 'utv', 'nht', 
'std', 'rx1', 'c70', '80f', 'ee5', 'fzs', 'awa', 'rx2', 'acf', 
'ax4', 'xlr', '17b', 'yfm',

'450f', '125f', 'f900', '125r', '200x',
'mt09', '150l', '250x', '160r', '250r', '890r', '502x', 'st-r',
'ttds', 'xadv', 'z900', 'n250', '900r', 'x150', 'tb-x', 'c600',
'250z', '190l', 'bwsx', '150x', 'aiii', 'n160', 't100', 'er6f',
'650r', 'c400', '310m', '250f', 'g310', 't120', 'd250', 'sx-f',
'790r', '200r', 'ybrz', '180s', '110f', '800f', 'p150', 'z250',
'200s', 'vrsc', 'r150', '650a', '155u', '110r',
'mt03', '250n', 'mt01', '700r', 'z400', '402s', 'z650', '310r',
'rm-z', 'fz8s', 'rsv4', 'zx6r', '110i', 'f800', 'fz25', 'c150',
'kx60', '302s', 'dr50', 'mt07', '650s', 'smcr', '660r', '100t', 
'fz16', '150r', '100r', 'c125', 'mt15', '350w',

'650mt', 'exc-f', '650dl', '1200z', '200xr', 'g310r', 'cb650', 
'570sp', '1000s', 'dr200', 'cb175', '850gs', '950sp', 'u1-l3', 
'tnt25', 'xt660', 'ex300', 'fz250', 'le300', 'ac300', '500fe', 
'r1300', '1200s', 'r1200', 'x-adv', 'nc750', 'plr70', 'gsx-s', 
'1000r', 'cb160', 'ttr50','zx10r', 'zx-6r', '180ug', '250sr', 
'450mt', 'gs750', '180gt', '250rx', 'v-rod', 'plr90', 'cr125', 
'929rr', '650xt', 'sr250', 'fxsts', 'fxdrs', 'bs200', '180fi', 
'250sx', 'nk250', 'r1250', 'r15v3', '800de', '150rb', 'tt125', 
'gs310', 'gs650', 'f900r', 'rs200',

'crf250', 'fz-150', 'zx-10r', 'cb650r', '1000-3', 'xre190', 
'rx-115', 'gsx125', '125-2a', 'cb125f', 'crf50f', 'af115s', 
'f750gs', 'f650st', 'fzn250', '1100nv', 'x-town', 'plr450', 
'xtz150', 'g650gs', 'x-advr', '1050de', 'fzs6vr', 'cb160f', 
'r1250r', 'adv750', 'r-nine', 'f900xr', 'nc750x', 'xr650l', 
's1000r', 'zr1000', 'cb190r', 's2r800', 'xr150r', 'pcx160', 
'ltz400', 'cbf150', 'plr250', 'rx-100', 'sz15rr', 'ra1250', 
'cb300f', 'gsx-8s', 'z250sl', 'cbr250', 'z1000b', 'xtz660', 
'1000rr', 'dcx88c', 'le650f', 'kle650', 'f700gs',

'xmax300', 'comm200', 'bwsx125', 'r1200gs', 'crf450r', 'klx150j', 
'cr4-162', 'adv750m', 'sv650/a', 'crf110f', 'aiii125', 'ltz-400',
'1000rrc', 'cb1000r', 'ktm890r', 'aiii150', 'ak125rl', 'fzn150d', 
'ct100es', 'rx300-4', '1000x-4', 'ay200zh', 'tnt150i', 'adv750k', 
'klz1000', 'crf150r', 'vt750dc', 'gsxr150', 'r1250gs', 'utv300x', 
'crf250r']


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
