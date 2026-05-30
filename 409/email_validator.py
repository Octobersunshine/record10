import re
import time
import socket
import asyncio
import smtplib
import dns.resolver
import dns.asyncresolver
from dataclasses import dataclass, field
from typing import Callable, Optional


MX_CACHE_TTL = 3600
MX_CACHE: dict[str, tuple[float, bool, list[str]]] = {}
SMTP_CACHE_TTL = 1800
SMTP_CACHE: dict[str, tuple[float, bool, str]] = {}
DNS_TIMEOUT = 3.0
SMTP_TIMEOUT = 5.0
MAX_CONCURRENT_DNS = 10
MAX_CONCURRENT_SMTP = 5


DISPOSABLE_EMAIL_DOMAINS = {
    "10minutemail.com", "10minutemail.net", "10minutemail.org",
    "temp-mail.org", "tempmail.com", "tempmail.net",
    "throwawaymail.com", "dispostable.com", "mailinator.com",
    "mailinator.net", "mailinator2.com", "mailinator.alt.com",
    "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "sharklasers.com", "grr.la", "spamgourmet.com",
    "spambob.com", "spambob.net", "spambob.org",
    "mintemail.com", "trashmail.com", "trashmail.net",
    "trashmail.org", "trashymail.com", "trashymail.net",
    "yopmail.com", "yopmail.net", "yopmail.fr",
    "getairmail.com", "disposable-email.com",
    "fakeinbox.com", "fakeinbox.net", "fakeinbox.org",
    "emailthe.net", "emailsensei.com", "bouncr.com",
    "emailtemporanea.com", "emailtemporanea.net",
    "nowmymail.com", "thisisnotmyrealemail.com",
    "killmail.com", "mailnesia.com", "mailcatch.com",
    "centermail.com", "centermail.net",
    "shortmail.net", "chammy.info",
    "0-mail.com", "027168.com", "0815.ru",
    "0wnd.net", "0wnd.org", "126.com.us",
    "140unichars.com", "1fsdfdsfsdf.tk",
    "1mail.ml", "1pad.de", "1s.fr",
    "20mail.it", "20mail.in", "21cn.com",
    "24hourmail.com", "2prong.com",
    "30minutemail.com", "33mail.com",
    "420blaze.it", "4email.net", "4warding.net",
    "5mail.net", "60minutemail.com",
    "7days-print.com", "7mail.ml",
    "8127ep.com", "8mail.net",
    "99.com.ru", "990000.ru",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk.com",
    "abv.bg", "ac20mail.it",
    "accessinternet.nl", "acespikes.com",
    "acmepcs.com", "acn-video.com",
    "activeprospect.com", "adexec.com",
    "adf.ly", "aditus.be",
    "adosia.com", "adrianogratieri.com.br",
    "aeqva.net", "afrobacon.com",
    "ag.us.to", "agedmail.com",
    "ahnibea.com", "aichoo.com",
    "aintinternet.com", "airshipp.com",
    "ak71.net", "al-qaeda.us",
    "alassasoftware.com", "albion2.com",
    "alepetro.net", "aligames.net",
    "alivance.com", "allgoodmail.com",
    "allmail.net", "allsunday.net",
    "alpenjodel.com", "alt-punk.com",
    "alternate-mail.com", "ama-trade.de",
    "amadeusrock.com", "amail.com",
    "amail4.me", "amateurbusiness.com",
    "amdalliance.com", "ameritech.net",
    "ams13.com", "anappfor.com",
    "andthen.us", "anime-ftw.net",
    "anonbox.net", "anonimg.com",
    "anonmail.asia", "anonmail.ch",
    "anonmail.dk", "anonmail.in",
    "anonmail.info", "anonmail.mobi",
    "anonmail.name", "anonmail.net",
    "anonmail.org", "anonmail.ru",
    "anonmail.ws", "anotherdomaincyka.tk",
    "anthony-melnikov.com", "antichef.com",
    "antireg.com", "antireg.ru",
    "antonovich.name", "aol.com.nyud.net",
    "aonmail.com", "apfelcast.de",
    "applelinks.net", "appriver.com",
    "aqute.biz", "aranock.org",
    "arcor.de", "arghmail.com",
    "arkleys.com", "armyspy.com",
    "arseiam.com", "artman-conception.com",
    "arurichika.tokyo", "aschenbrandt.com",
    "asian-sexuality.com", "ask-mail.com",
    "astechonline.com", "asu.edu",
    "at0mik.org", "atcpost.com",
    "ateljeoteker.com", "atoomail.com",
    "att.net", "audiomessage.net",
    "auglink.com", "ausgezeichnet.org",
    "automail.ai", "autosender.ru",
    "avansigorta.com", "aver.com",
    "avpa.info", "awwwards.me",
    "ayomail.com", "azcomputertek.com",
    "azhuncang.com", "b1n.net",
    "b2cmail.de", "baddi.net",
    "badgerlandbrew.com", "bahnhof.se",
    "bajomp3.com", "balkanweb.com",
    "ballsofsteelalliance.com", "bamail.net",
    "bancolombia.com.co", "bandtshirts.com",
    "barryogorman.com", "baxomale.ht.cx",
    "bbhost.info", "bccto.me",
    "bcmpropertyservices.com", "bdm-designs.com",
    "bearsarefuzzy.com", "beddly.com",
    "beefmilk.com", "beer.com",
    "bell.net", "bellaliant.net",
    "bendixon.com", "benipaul.com",
    "benthegeek.net", "berea.edu",
    "berlin-koch.com", "bestchoiceforgirls.info",
    "bestemailusa.com", "bestlibrary.net",
    "bestmail.us", "bestvpnrating.com",
    "betheadvocate.com", "bettysgonedoggie.com",
    "beud.net", "bfcsi.org",
    "bftps.com", "bgv.bg",
    "bhuz.com", "bibme.org",
    "bidourl.com", "big1.us",
    "bigstring.com", "bigwhoop.co.za",
    "bike-adventure.com", "billstomail.com",
    "bin.gy", "binhost.de",
    "bio-massa.info", "biwmail.com",
    "biyacg.com", "bjeee.com",
    "bk.ru", "blackmarket.to",
    "bladesmail.net", "bleh.com",
    "blip.ch", "blockexec.com",
    "blog.com", "blogmyway.org",
    "blogos.com", "blogspam.ro",
    "bluehome.net", "bluewin.ch",
    "boack.com", "boatmail.us",
    "bobanch.com", "bobmail.info",
    "bodesstore.com", "bofthew.com",
    "bollywood-host.com", "boo-mail.com",
    "bookthemmore.com", "bootybay.de",
    "borged.com", "borgel.com",
    "boun.cr", "bounceme.net",
    "boxformail.in", "boximail.com",
    "boylesoftware.com", "brapmail.com",
    "breaze.com", "brefmail.com",
    "briefemail.com", "broadbandninja.com",
    "brokenseal.org", "brooksmail.org",
    "brownies.com", "brucemakers.com",
    "bst7.net", "btmaz.com",
    "bucksheemail.com", "budaya-tradisional.com",
    "buffalobillszone.com", "bugfoo.com",
    "bugmenot.com", "bullshit.org",
    "bumpytruck.net", "bunnybait.com",
    "burger-mail.com", "burmacomputers.com",
    "burnthespam.com", "bus-iberia.com",
    "business-switzerland.com", "buy-accutane24.com",
    "buy24h.com", "buymoreplays.com",
    "bxnb.com", "byebyemail.com",
    "byespm.com", "c2.hu",
    "caich2.com", "callian.com.tw",
    "cam4you.cc", "camgirlzotica.com",
    "camosh.com", "campodebelgrano.com",
    "canada.com", "cancer8.com",
    "candw.net", "canelacinemas.com",
    "cannabis-seed-depot.com", "canoemail.com",
    "cantmix.com", "captain-hammer.com",
    "care2.com", "caring.com",
    "carinsurancemax.co.uk", "carls.co.za",
    "cars2sale.co.za", "cartelera.org",
    "cashette.com", "caspermail.com",
    "catbox.moe", "cates.net",
    "cbair.com", "cbf1.com",
    "cc.li", "ccmail.uk",
    "cdpa.cc", "cedant.com",
    "celticwarriors.net", "centenariomail.com",
    "certain.com", "cfego.com",
    "cfire.net", "cfosharing.com",
    "cgmail.com", "chacuo.net",
    "chammy.info", "cheaphub.com",
    "cheetmail.com", "chem.purdue.edu",
    "cherry.bz", "chez.com",
    "chiaramail.com", "chickenkiller.com",
    "china-job.de", "chipia.net",
    "choicemail1.com", "chong-mail.com",
    "chorebus.com", "christianmail.biz",
    "christianlouboutinshoes2015.us", "christunited.com",
    "chuckmail.com", "cinder411.com",
    "cinnamonroll.com", "circuitcityofsalem.com",
    "citusinmobiliaria.com", "ckiso.com",
    "cl-cl.org", "clandestino.co",
    "classicalmusicmp3.net", "claudiomatias.net",
    "claymania.com", "cleanmail.ch",
    "cleverbot.com", "clickdeal.co",
    "cliffordchance.com", "clipmail.eu",
    "clix.pt", "cloakedmail.com",
    "clothesencounters.net", "club386.com",
    "cluxx.com", "cmail.club",
    "cmail.nu", "cnmsg.net",
    "cnn.com.nyud.net", "cobragold.com",
    "cocovapor.com", "codepad.org",
    "coeus-solutions.de", "coffee-and-pie.com",
    "cojones.com", "colafanta.com",
    "coldemail.info", "comcast.net",
    "comhem.se", "comic.com",
    "communistindia.org", "compareshippingrates.com",
    "compu-util.com", "comwest.de",
    "conel.com", "confetti.nl",
    "confused.eu", "conocochauraman.com",
    "conservative.com", "contactamericans.net",
    "contactoffice.nl", "coolemail.biz",
    "coolimpool.org", "coonass.org",
    "coop.co.uk", "copernic.com",
    "copykat.com", "corbata.es",
    "corbinkitchen.com", "corecom.com",
    "cornell.edu", "correctness.nl",
    "cortlandt.com", "cosmorph.com",
    "coubic.com", "countycomm.com",
    "courtlistener.com", "coza.biz",
    "cpac.com", "cpan.org",
    "craiglist.org", "crapmail.org",
    "crawfishking.com", "crazespaces.com",
    "creditcard4u.co.uk", "creepypasta.com",
    "criminalia.net", "criticbay.com",
    "crotonaromatics.com", "crowdcrafting.org",
    "csh.rochester.edu", "csu.edu",
    "ctos.ca", "ctscanservices.com",
    "cube0.com", "cucumail.com",
    "cust.in", "cuu.be",
    "cwazy.co.uk", "cx.ua",
    "cyber-phone.eu", "cybergoth.biz",
    "cyber-mage.com", "cyberspymail.com",
    "cybo.net", "cylab.org",
    "d3p.dk", "dab.ro",
    "dack.com", "dadwork.com",
    "daemon-mail.com", "dallashotelcollection.com",
    "damaij.com", "damndelicious.net",
    "dancingsalsa.net", "dandik.com",
    "dapodik.com", "darkharvestfilm.com",
    "darkhorizons.com", "darlingmail.com",
    "dataarca.com", "datazo.ca",
    "davidlaines.com", "davmail.net",
    "dbunker.com", "dcemail.com",
    "dcard.tk", "dcccd.edu",
    "dcemail.com", "deadaddress.com",
    "deadchildren.org", "deagot.com",
    "dealja.com", "debtdoctor.com",
    "decaying.com", "dedicatetoyou.com",
    "deepdiscountdvd.com", "defendsemi.com",
    "dejanews.com", "delanoticias.com",
    "deliberateobfuscation.com", "delikkt.de",
    "delicious.com", "deliv.com",
    "deltacom.com", "demexp.org",
    "democracynow.org", "demon.co.uk",
    "dentakids.com", "despam.com",
    "despammed.com", "detroitglasscompany.com",
    "dev-null.cjb.net", "develop-drupal.com",
    "df7.com", "dfgh.net",
    "dharmatel.net", "dialogone.com",
    "diamonds.net", "dicksuckersanonymous.com",
    "dietobesity.com", "digitaltuxedo.com",
    "digi-hat.com", "dildoking.com",
    "dingbone.com", "diod.com",
    "directwholesalemeats.com", "dircon.co.uk",
    "discard-email.cf", "discard.email",
    "discard-email.ga", "discard-email.gq",
    "discard-email.ml", "discard-email.tk",
    "disign-concept.eu", "disktouch.info",
    "disposable.address.yopmail.com",
    "disposableinbox.com", "dispostable.com",
    "divismail.ru", "divermail.com",
    "dlemail.ru", "dmxsys.com",
    "dodgit.com", "dodgit.org",
    "dodsi.com", "domozmail.com",
    "dontmail.me", "dontsendmespam.de",
    "door-to-door.com", "dotroll.com",
    "douchemail.com", "downwiththesystem.org",
    "doxcom.net", "doxycycline-24.com",
    "droidnation.net", "drumcode.se",
    "drumperium.com", "dsgtricks.com",
    "dtnl.net", "dubli.com",
    "duckling.org", "dudmail.com",
    "dumpsv.com", "dustyfile.com",
    "dvfsaz.com", "dyceroprojects.com",
    "dynacooper.com", "dynamicsignal.com",
    "e-mail.com", "e4ward.com",
    "easy-trash-mail.com", "easytrashmail.com",
    "eatme.com", "ebeschluss.com",
    "ebizmba.com", "eclass.cuhk.edu.hk",
    "ecolo.org", "econe.biz",
    "edgex.ru", "edinburghbusinesslist.com",
    "ednewton.net", "edu-sa.org",
    "ee2.biz", "efxsucks.com",
    "eghc.org", "einmalmail.de",
    "einrot.com", "eintagsmail.de",
    "eldred.com", "elearningpost.com",
    "electro.mn", "elitedesigner.biz",
    "ellinea.com", "elmerdigital.com",
    "elvis.com", "elwoz.com",
    "embracetherock.com", "emergencyemail.org",
    "emkei.cz", "eml.pp.ua",
    "email-fake.com", "email.googlegroups.com",
    "emailigo.de", "emailias.com",
    "emailinfive.com", "emaillime.com",
    "emailmiser.com", "emailproxsy.com",
    "emailresort.com", "emails.ga",
    "emailsensei.com", "emailspam.cf",
    "emailspam.ga", "emailspam.gq",
    "emailspam.ml", "emailspam.tk",
    "emailt.com", "emailtmp.com",
    "emailto.de", "emailwarden.com",
    "emailx.at.hm", "emailxfer.com",
    "emeil.in", "emetab.com",
    "emil.com", "emz.net",
    "enayu.com", "encr.pw",
    "endlesspe.com", "engaged.com",
    "engmail.com", "ensignnet.com",
    "entertainment.com", "ephemail.net",
    "epicidiots.com", "eqiluxmail.com",
    "ericjohnson.org", "es.gy",
    "escapemail.info", "esem.es",
    "esgeneri.com", "essimail.com",
    "etranquil.com", "etranquil.net",
    "etranquil.org", "eudoramail.com",
    "everytg.com", "evanfox.com",
    "evilstorm.org", "ewide.net",
    "excaliburfilms.com", "excitem.com",
    "exemail.com.au", "execulink.com",
    "exin.xyz", "explodemail.com",
    "express.net", "eyepaste.com",
    "ezclixx.info", "ezgig.net",
    "ezite.ro", "f262.com",
    "facebook-email.cf", "facebook-email.ga",
    "facebook-email.gq", "facebook-email.ml",
    "facebook-email.tk", "facebookmail.com.nyud.net",
    "fackme.ga", "falconmail.com",
    "fammix.com", "fan-club.org",
    "fandeluxe.com", "fansworldwide.de",
    "fantasymail.de", "fastacura.com",
    "fastchevy.com", "fastchrysler.com",
    "fastkawasaki.com", "fastmazda.com",
    "fastmitsubishi.com", "fastnissan.com",
    "fastsubaru.com", "fastsuzuki.com",
    "fasttoyota.com", "fastyamaha.com",
    "fatflap.com", "fbi.gov.nyud.net",
    "fbmail15.com", "fcg.com",
    "femail.com", "fernweh-reise.com",
    "fetchmail.co.uk", "fictionalemail.com",
    "fidmail.com", "fightallspam.com",
    "fiifke.de", "file2mail.com",
    "filzmail.com", "fir.org",
    "fishmail.de", "fitnessboutique.co.uk",
    "fivemail.de", "fixmail.tk",
    "fizmail.com", "flabot.com",
    "flakyemail.com", "flashbang.ca",
    "flexmls.com", "floridalotteryresults.org",
    "flowu.com", "fluffymail.com",
    "flux.com", "fly-ts.de",
    "flyinggeek.net", "fmf.mx",
    "fmail.co.uk", "foggymail.co.uk",
    "fontlord.com", "foodbooth.net",
    "football.ua", "foreverconscious.com",
    "fornow.eu", "forward.cat",
    "foxtrotter.info", "fr33mail.info",
    "fragolina22.com", "frankfurter-it.de",
    "frasq.com", "fraud.tk",
    "freakygirls.net", "free-email.cf",
    "free-email.ga", "free-email.gq",
    "free-email.ml", "free-email.tk",
    "freeblackberrystormthemes.com", "freecat.net",
    "freedompop.com", "freegamelots.com",
    "freehotmail.org", "freeletter.me",
    "freemail.ms", "freemails.cf",
    "freemails.ga", "freemails.gq",
    "freemails.ml", "freemails.tk",
    "freemeil.ga", "freemeil.gq",
    "freemeil.ml", "freemeil.tk",
    "freenet.de", "freeola.com",
    "freepopping.com", "freeporndb.us",
    "freescale.com", "freeshell.org",
    "freesmtpservers.com", "freespf.com",
    "freeurl.fr", "freewebemail.net",
    "fremmed.no", "fresnel.fr",
    "freundin.ru", "friendmoy.com",
    "fromru.com", "fsidaho.net",
    "fuckingduh.com", "fudgerub.com",
    "fuirio.com", "fullarmor.com",
    "fumo.info", "fun64.com",
    "funkchemistry.com", "funpic.org",
    "funtime.com", "furzauflunge.de",
    "future-mx.com", "fux0ringduh.com",
    "fw.by", "fwhh.de",
    "g1l.com", "gafy.net",
    "gamil.com", "gamezoo.com",
    "garliclife.com", "garysims.net",
    "gawab.com", "gearhost.com",
    "gemein.com", "generalmail.com",
    "genick.com", "gentlemint.com",
    "georgerrmartin.com", "geschent.biz",
    "get-mail.cf", "get-mail.ga",
    "get-mail.gq", "get-mail.ml",
    "get-mail.tk", "get2mail.fr",
    "getairmail.com", "getemail.website",
    "getinbox.eu", "getmails.eu",
    "getnada.com", "getonemail.com",
    "getonemail.net", "getpe.eu",
    "ghc.net", "ghosttexter.de",
    "giacomo.biz", "giftbuzz.com",
    "giga.ovh.org", "gmx.com",
    "go2usa.info", "goat.si",
    "goblin.homedns.org", "godut.com",
    "goingnowhere.org", "golemico.com",
    "goodmail.com", "goodyearsouthtylerville.com",
    "googile.com", "googlemail.com.nyud.net",
    "gopostal.info", "gordano.com",
    "gothic.net", "gov.gy",
    "gqz.nu", "graffiti.net",
    "grandmamail.com", "grandmasmail.com",
    "great-host.in", "greggirard.com",
    "gridview.net", "grishnak.com",
    "groupbuff.com", "grosche.ca",
    "grownupgeek.com", "gruppotv.com",
    "gsrv.co.uk", "gtxhome.com",
    "guerillamail.biz", "guerillamail.com",
    "guerillamail.net", "guerillamail.org",
    "guerrillamail.biz", "guerrillamail.com",
    "guerrillamail.de", "guerrillamail.info",
    "guerrillamail.net", "guerrillamail.org",
    "guerrillamailblock.com", "guessmail.info",
    "gustr.com", "h8s.org",
    "habmalnefrage.de", "hackthatbit.com",
    "hatespam.org", "hcat.edu",
    "headstrong.de", "heathenhammer.com",
    "heathernova.net", "hecth.com",
    "hellodream.mobi", "helpinghandtaxcenter.org",
    "herpderp.nl", "heypix.net",
    "hh-64.de", "hidemail.de",
    "hidemail.us", "highbros.com",
    "hm-temp.com", "home.dk",
    "home.nl", "homesbuilt.net",
    "homesellergate.com", "hottempmail.com",
    "housat.com", "howtogame.com",
    "hr84.com", "hsdut.com",
    "hu242.com", "hukkmu.com",
    "hulu.com.nyud.net", "hungry.com",
    "hush.ai", "hushmail.com.nyud.net",
    "hushmail.me", "hyf.de",
    "hyper107.com", "i2pmail.org",
    "iabuz.com", "icanhaz.com",
    "icxnu.com", "id.au",
    "idler411.com", "ieatspamforbreakfast.com",
    "ifrance.com", "iiez.com",
    "iguanademail.com", "iheart.com.nyud.net",
    "ikbenspamvrij.nl", "illinois.edu",
    "imails.info", "imgv.de",
    "imgvault.ca", "imstations.com",
    "in-box.su", "in-mail.be",
    "inbax.tk", "inbox.si",
    "inboxalias.com", "inboxbear.com",
    "inboxclean.com", "inboxclean.org",
    "inboxed.nl", "inboxhero.com",
    "inboxlit.com", "inboxproxy.com",
    "inboxstore.org", "incog-mail.eu",
    "indeedemail.com", "indigo.ie",
    "indiatimes.com", "indosec.net",
    "inertiant.com", "info-radio.de",
    "info-rmx.com", "infocom.zp.ua",
    "infonie.fr", "internet-viet.de",
    "internetmail.com", "interstats.org",
    "intopwa.com", "inuit.ca",
    "inutil.com", "io.ua",
    "ip6.li", "ipoo.org",
    "ipsmartz.net", "iqemail.com",
    "irania.tk", "ironie.ch",
    "is.af", "ischool.berkeley.edu",
    "isna.info", "isp.com",
    "italia.it", "iubilaeum.org",
    "iusearchbtw.com", "ivoryandgoldwedding.com",
    "ixaxaar.com", "j299.com",
    "jabb.cz", "jabmail.co.uk",
    "jamesbondisaloser.com", "janproen.com",
    "jdownloader.org", "je-recycle.info",
    "jeh-circuit.com", "jellii.com",
    "jmail.com", "jmail.co",
    "johnnybit.com", "joinfacebook.org",
    "jopho.com", "josefek.com",
    "jourrapide.com", "junk1e.com",
    "junkmail.ga", "junkmail.gq",
    "junkmail.ml", "junkmail.tk",
    "justemail.ml", "kasmail.com",
    "kaspop.com", "keepmymail.com",
    "keemail.me", "kennynet.nl",
    "keromail.com", "kfchickenrice.com",
    "kgreen.net", "khmerforums.com",
    "killmail.net", "kimsdisk.com",
    "kingstonsja.com", "kissfans.ru",
    "kittymail.com", "klassmaster.com",
    "klipschxlife.com", "kloap.com",
    "kmail.com", "kode.im",
    "koin.me", "kom.pl",
    "koszmail.pl", "kryptonpassword.com",
    "kuhrap.com", "kulturbetrieb.org",
    "kulturmail.com", "kumail.com",
    "l33r.eu", "labetteraverite.fr",
    "lacedmail.com", "lacroixshow.com",
    "lafontaine.ca", "lakdiva.org",
    "land.ru", "larryhall.ugu.pl",
    "laszlopinter.com", "last-chance.pro",
    "lavabit.com", "lawlita.com",
    "lazyinbox.com", "ldinternet.com",
    "leeching.net", "lek.no",
    "lentinemarine.com", "letmehavemail.com",
    "leechbug.com", "legalrc.com",
    "lehmanmail.com", "lenin.info",
    "leo.org", "lerctr.org",
    "letmeinonthis.com", "letsmailme.de",
    "lez.se", "lgxscreen.com",
    "lhmssucks.com", "libero.it",
    "liebemal.de", "lifebyfood.com",
    "lillemaposten.no", "limboprogress.com",
    "lineone.net", "link2mail.net",
    "linuex.com", "linuxmail.so",
    "linuxaddicted.com", "lionsmane.com.au",
    "liquivida.com", "list.ru",
    "litedrop.com", "littlegremlin.com",
    "live.ca", "live.co.uk",
    "live.com", "live.de",
    "live.fr", "live.nl",
    "liveradio.com", "lkgn.org",
    "llogin.ru", "loadaveragezero.com",
    "locanto.com", "lockmail.com",
    "logisticsteam.com", "london.com",
    "lordsofts.com", "loseyourip.com",
    "lovebeautyplanet.com", "lowtek.ca",
    "lroid.com", "lr7.us",
    "lsec.cc.ac.cn", "ltg email.com",
    "luckymail.org", "luv2.us",
    "lyfestylemarket.com", "lycos.com",
    "lyricsxp.com", "m-hmailer.info",
    "m21.cc", "m4ilweb.info",
    "ma1l.biz", "ma1l.ga",
    "macromates.com", "maenner.info",
    "magsphoto.com", "maileater.com",
    "mailexpire.com", "mailinator2.com",
    "mailinator.co.uk", "mailinator.com",
    "mailinator.net", "mailinator.org",
    "mailinator.us", "mailinater.com",
    "mailinate.com", "mailinblack.com",
    "mailincubator.com", "mailismagic.com",
    "mailme.gq", "mailme.icu",
    "mailme.lv", "mailmetrash.com",
    "mailmoat.com", "mailms.com",
    "mailnator.com", "mailnesia.com",
    "mailnull.com", "mailpick.biz",
    "mailpluss.com", "mailpooch.com",
    "mailps.de", "mailqw.net",
    "mailru.com", "mailsac.com",
    "mailscrap.com", "mailshell.com",
    "mailsiphon.com", "mailslapping.com",
    "mailspam.xyz", "mailtemp.info",
    "mailtome.de", "mailtothis.com",
    "mailtrash.net", "mailtv.tv",
    "mailzilla.com", "mailzilla.org",
    "makemetheking.com", "malahov.de",
    "malayalamdirectory.com", "malverncomputers.com",
    "mamba.com", "mangocandy.com",
    "manifestconference.net", "mapremier.com",
    "marshut.com", "martinique.org",
    "mary24.com", "masrawy.com",
    "master-mobel.de", "matchpol.com",
    "matsuri.fr", "mawode.com",
    "max-mail.us", "maxigame.biz",
    "mazhar.de", "mbx.cc",
    "mcdonalds.com.nyud.net", "mcut.be",
    "md-portal.com", "me.com",
    "meb.gov.tr", "medphys.mcw.edu",
    "meepsheep.com", "mega-otaku.com",
    "meinspamschutz.de", "meld.de",
    "melty.fr", "memail.com",
    "memorybox.info", "merry.pink",
    "meshpromotion.com", "metservice.com",
    "metacrawler.com", "metamorf.com",
    "mewsing.org", "mfsa.ru",
    "mh-zwei.de", "mi5.gov.uk.nyud.net",
    "miaferrari.com", "microsoft.com.nyud.net",
    "midcoastcustoms.com", "midcoastcontrols.com",
    "midnightbox.com", "mietmail.de",
    "miiii.ru", "milfme.com",
    "milivojevic.com", "minibox.email",
    "minniewest.com", "mintemail.com",
    "misterpinball.de", "miva.com",
    "mixcat.com", "mji.mobi",
    "ml1.net", "ml8.ca",
    "mm.st", "mnsi.net",
    "moakt.co", "moakt.com",
    "mobileninja.co.uk", "moburl.com",
    "moe.hm", "moimoin.de",
    "moldova.org", "mollerlawgroup.com",
    "moncourriel.fr", "monibot.com",
    "monumentmail.com", "moozi.co",
    "mopsite.com", "moreorcs.com",
    "morningtoast.com", "mortality.net",
    "mosaicdesigns.co.nz", "motique.com",
    "mountainregion.com", "mouse-potato.com",
    "msa.minsal.cl", "msn.com.nyud.net",
    "mt2009.com", "mtinternet.com.br",
    "muchomail.com", "mucura.com",
    "muffinja.com", "mulf.org",
    "mundowap.com", "mvrht.com",
    "mvrht.net", "my.af",
    "my10minutemail.com", "mybitti.de",
    "mycleaninbox.net", "mycorneroftheinternet.net",
    "myde.ml", "myemailboxy.com",
    "myindohome.com", "myinterserver.com",
    "mymail-in.net", "mymail8.com",
    "mymailo.com", "mynet.com",
    "mynicesite.com", "mynokiablog.com",
    "myopang.com", "myspace.com.nyud.net",
    "myspamless.com", "mystvpn.com",
    "mytemp.email", "mytempmail.com",
    "mythrash.com", "mytrashmail.com",
    "mywrath.net", "myyearbook.com.nyud.net",
    "n29.com", "naver.com.nyud.net",
    "nbox.org", "nctr.edu",
    "neomailbox.com.nyud.net", "nepw.com",
    "nervhq.org", "net4india.com",
    "netaddress.com", "netbounce.de",
    "netcourrier.com", "netimail.fr",
    "netmails.com", "netscape.com",
    "netscapeonline.co.uk", "nettaxi.com",
    "netzero.com", "netzero.net",
    "neuf.fr", "newmail.net",
    "newwavecon.com", "nexxmail.com",
    "nfast.com", "ngs.ru",
    "nhc.noaa.gov.nyud.net", "nice-4u.com",
    "nickstel.com", "nie-wiem.tk",
    "nikhef.nl", "nincsmail.com",
    "nintendo.com.nyud.net", "noblepioneer.com",
    "nobugmail.com", "noicd.com",
    "nokiablog.com", "nomorespam.com",
    "nonspam.eu", "nospam.ze.tc",
    "nospam4.us", "nospamfor.us",
    "nospamthanks.info", "notmailinator.com",
    "notsharingmy.info", "nowhere.org",
    "nowmymail.com", "ntlworld.com",
    "nullbox.info", "nuts2ua.com",
    "nutrirain.com", "nuvs.net",
    "nwlink.com", "nycmetro.com",
    "nyrr.org", "nzz.ch",
    "o2.pl", "o2online.de",
    "o7i.net", "oakmail.net",
    "obfusko.com", "obobbo.com",
    "obsob.com", "odaymail.com",
    "odermark.com", "offshoreproxies.net",
    "oftw.net", "okclips.net",
    "okrent.net", "olimpo.tk",
    "olivetti.net", "omail.pro",
    "one-time.email", "one2mail.info",
    "oneoffemail.com", "oneoffmail.com",
    "online.ms", "onqmarketing.com",
    "onread.com", "ontherock.ca",
    "oopi.org", "opayq.com",
    "opel.com.nyud.net", "open-mail.info",
    "openr.com", "opentrash.com",
    "op.pl", "orange.fr",
    "orange.net", "orcon.net.nz",
    "oreidafjoll.com", "oriental-cuisine.de",
    "originalmail.info", "oriole-mail.com",
    "ourklips.com", "outgun.com",
    "outlook.com.nyud.net", "outmail.info",
    "ovh.net", "owlpic.com",
    "oxopolitics.com", "ozlabs.org",
    "p71.de", "pancakemail.com",
    "paplease.com", "parliment.gov.uk.nyud.net",
    "partybot.org", "paulconnell.com.au",
    "paypall.com.nyud.net", "pcusers.otherwhen.com",
    "pe.hu", "pege.hu",
    "penisgoes.in", "pereiraedt.com.br",
    "persecond.info", "personal.gr",
    "pfui.com", "pgl.yoyo.org",
    "phpbb.ug", "phreaker.net",
    "pica-pau.com.br", "picmail.de",
    "pimpedupmyspace.com", "pine.com",
    "pinggg.com", "piratil.com",
    "pizza.ww7.ca", "plaf.org",
    "planet-inter.com", "plasticmans.com",
    "playful.net", "pleasenospam.com",
    "plhk.net", "plopsite.be",
    "plu.eu", "plumber24h.com",
    "plw.me", "plusmail.info",
    "pmail.net", "pnhl.com",
    "pocketmail.com", "pogo.com.nyud.net",
    "poise.com", "politikerclub.de",
    "polyopia.com", "pomail.net",
    "pongonova.com", "pontiac.3300i.net",
    "popcorn.email", "poppycorn.com",
    "porn-hot.com", "pornorod.com",
    "portsmouth.com", "post.com.nyud.net",
    "post4.pl", "postfach.tk",
    "postmaster.net", "poutine.biz",
    "poverty.ac.uk", "powered.name",
    "powweb.com", "pp.ht",
    "praisedegree.com", "pratik.com.bd",
    "primalware.co.uk", "primus.com.au",
    "privacy.net", "private-mail.cf",
    "private-mail.ga", "private-mail.gq",
    "private-mail.ml", "private-mail.tk",
    "privatdemail.net", "privy-mail.com",
    "probomail.com", "prodigy.net",
    "profusion.com", "programer.net",
    "proinbox.com", "projectxonline.com",
    "prolan.com", "promessage.com",
    "pronmail.com", "protempmail.com",
    "protonmail.ch", "protonmail.com",
    "pruconnect.net", "ps3portal.com",
    "pseudo.net", "pshift.com",
    "psychz.net", "pt.lu",
    "ptrmail.com", "publikusz.sk",
    "pulsejet.co", "punkass.com",
    "purelogistics.org", "pushauction.com",
    "put2.net", "putthisinyourspamdatabase.com",
    "pwrby.com", "q5vm.com",
    "qapo.com", "qbbit.com",
    "qei.biz", "qkrq.de",
    "qms.nu", "qsp.thc.so",
    "quadrafit.com", "qualmail.com",
    "queerfilmfestival.com", "quickemail.info",
    "quickinbox.com", "quickmail.nl",
    "quotaless.com", "qwghlm.co.uk",
    "r-a-n-d-o-m.com", "r4nd0m.info",
    "rabin.com", "rabinowitzcommunications.com",
    "racing.hu", "radikaldialektik.com",
    "rafmail.com", "raketmail.ch",
    "rambler.ru", "rasmsen.org",
    "rattintrash.com", "ravensburg.de",
    "rawbw.com", "rbcmail.com",
    "rcaschool.org", "rdc.to",
    "realtyexecutivesmi.com", "recursor.net",
    "recyclemail.dk", "reddwarf.dk",
    "reflexion.net", "regebro.com",
    "regb.tk", "rejectmail.com",
    "reliable-mail.com", "reloadmedia.com",
    "remail.zone", "rememberry.com",
    "renault.com.nyud.net", "repairingwindows.net",
    "replaymine.com", "reporting.net",
    "revolvingdoor.info", "revolvy.com",
    "rf3.net", "rg.net",
    "rhythm.net", "ricardorodrigues.com",
    "ride-swim.com", "rilomail.com",
    "riming.org", "risingsuntimes.com",
    "risus.org", "rj11.net",
    "rklb.de", "rkompass.com",
    "rma-online.co.uk", "rnd.de",
    "roadrunner.com", "robertocpa.com",
    "robthepeopleselfstorage.com", "rock.com",
    "rocketmail.com.nyud.net", "rogers.com",
    "ronn.nl", "rootfest.net",
    "rootshell.be", "royal.net",
    "rpgi.it", "rrr.com",
    "rt.tc", "rudolf-ptacek.de",
    "ruffrey.com", "ruky.id.au",
    "runbox.com", "rushpost.com",
    "ruthie.com", "s0ny.net",
    "s33dmail.com", "sabrestlouis.com",
    "safe-mail.net", "safersignup.de",
    "safetymail.info", "sahyogyaan.org",
    "salatsalat.de", "sameh.ae",
    "sandwich.net", "sangoma.net",
    "sasknet.sk.ca", "sbcglobal.net",
    "scalemail.info", "schamail.de",
    "schrott-email.de", "science.de",
    "scisla.com", "scn-inc.com",
    "scooby.org", "scottclugston.com",
    "scrapbook.com", "screenerd.com",
    "scsolicitors.com", "sdiy.cn",
    "se-avocats.fr", "seanmadden.com",
    "searchlo.com", "secretemail.de",
    "secure-mail.biz", "secure-mail.cc",
    "secureroot.com", "seed-mail.com",
    "seemyroom.com", "send-free-fax.com",
    "sendfreefax.net", "sendthis.com",
    "serialmail.com", "server.ms",
    "sessomic.com", "seznam.cz",
    "sfr.fr", "shabelle.com",
    "shadowmere.ws", "shaw.ca",
    "shcomp.org", "shesto.com",
    "shieldemail.com", "shiftmail.com",
    "shikakuteki.net", "shinymail.info",
    "shitaway.com", "shitmail.me",
    "shitmail.org", "shmer.com",
    "shogol.com", "shopping.com.nyud.net",
    "shortcircuit.info", "shortmail.net",
    "shotmail.ru", "showeb.net",
    "shturm.ru", "sibmail.com",
    "sify.com", "sift.co.uk",
    "silverwraith.com", "simail.ir",
    "simgal.com", "sina.com.nyud.net",
    "sinnlos-mail.de", "sions.com",
    "sis.com.uy", "sitemail.org",
    "skynet.be", "slapsfromthesky.com",
    "slave-tothe-box.net", "sloemail.com",
    "slothmail.net", "slowfood.de",
    "slushio.com", "smart.apm.pl",
    "smap.2y.net", "smapfree.2y.net",
    "smellfear.com", "smellslikepizza.com",
    "smtp.bz", "snafu.de",
    "snappymail.ca", "sneakemail.com",
    "sniksnak.nl", "snkmail.com",
    "soborgmassage.com", "social-mailer.com",
    "socialfurry.com", "socialnerds.org",
    "softhome.net", "sofimail.com",
    "sogetthis.com", "soisz.com",
    "solar-impact.co.uk", "soldat.ru",
    "solutions-4-me.info", "solvemail.info",
    "sogoweb.biz", "somuchradio.com",
    "songfestival.org", "sony.com.nyud.net",
    "sorted.biz", "sound-machine.net",
    "southbaytattoo.com", "sp4m.us",
    "spacebattles.com", "spam.la",
    "spam.su", "spamail.de",
    "spamarrest.com", "spamaway.net",
    "spambob.com", "spambog.com",
    "spambog.net", "spambog.ru",
    "spamcannon.com", "spamcatch.org",
    "spamcero.com", "spamcon.org",
    "spamcorptastic.com", "spamcowboy.com",
    "spamday.com", "spamex.com",
    "spamfighter.com", "spamfree.eu",
    "spamfux.com", "spamgoblin.com",
    "spamhole.com", "spamify.com",
    "spaminator.de", "spamming.info",
    "spaml.com", "spammotel.com",
    "spamoff.de", "spamorph.com",
    "spamover.net", "spamsalad.info",
    "spamslice.com", "spamspot.com",
    "spamstack.net", "spamthis.co.uk",
    "spamthisplease.com", "spamtrap.ro",
    "spamtroll.net", "speedgauge.net",
    "spidermonkey.com.ar", "spikio.com",
    "spoofmail.de", "sport.rr.com",
    "spraypainter.com", "squeakywheel.net",
    "squid-mail.com", "srifamily.com",
    "ssl-mail.com", "starlight-breath.net",
    "starpower.net", "stashmail.com",
    "stccommunications.com", "steaknshake.com.nyud.net",
    "steenbok.nl", "stinkefinger.net",
    "stop-my-spam.com", "stoopid.net",
    "storiqasucks.com", "stormloader.com",
    "stp6.com", "strefa.pl",
    "str3et.com", "streaming.bz",
    "studiocoast.com.au", "stupidforex.com",
    "subirimagenes.com", "suckmyd.com",
    "sucuri.net", "sudolife.com",
    "suffernomore.info", "sui-sin.com",
    "sumail.com", "sunblock.net",
    "super-auswahl.de", "supergreatmail.com",
    "supermailer.jp", "superplatform.biz",
    "superstachel.de", "suremail.info",
    "susi.be", "sustainabilitymatters.net",
    "sverigesradio.se", "svitno.com",
    "swcp.com", "sweetcherrypie.com",
    "swimchina.com", "swissinfo.org",
    "syntheticmedia.net", "syujob.com",
    "szacun.net", "t8k.de",
    "tadaa.de", "tagmymedia.com",
    "takethatmail.com", "taltio.com",
    "tamiltorrents.net", "tapchicuoi.vn",
    "task2.biz", "tasteit.com.ve",
    "taxi-london.com", "taylor-mail.com",
    "tdctrade.com", "techemails.com",
    "techfloyd.com", "technocage.com",
    "techportal.in", "teecrash.com",
    "teewars.org", "telecomsathome.com",
    "telefunken.com", "teleworm.com",
    "temp-mail.org", "temp-mail.ru",
    "tempalias.com", "tempe-mail.com",
    "tempemail.biz", "tempemail.co.za",
    "tempemail.net", "tempinbox.co.uk",
    "tempinbox.com", "tempmail.co",
    "tempmail.eu", "tempmail.fo",
    "tempmail.it", "tempmail.net",
    "tempmail.nyc", "tempmail.pt",
    "tempmail.us", "tempomail.fr",
    "temporarily.de", "temporarioemail.com.br",
    "temporaryemail.net", "temporaryemail.us",
    "temporaryforwarding.com", "temporaryinbox.com",
    "tempsky.com", "tempthe.net",
    "tempymail.com", "tenminutesmail.com",
    "tenvil.com", "tepid.org",
    "terminalmail.us", "tesco.net",
    "test.com", "testudine.com",
    "text2day.com", "tfet.net",
    "thankyou2010.com", "thc.so",
    "thedailytube.com", "thefacebook.com.nyud.net",
    "thegema.com", "thegr8skate.com",
    "theinternetemail.com", "thelema-club.org",
    "themudflap.com", "themtx.com",
    "thenospam.com", "theonion.com.nyud.net",
    "thepope.com", "theredears.com",
    "theteast.com", "theworstof.tv",
    "thinkbots.net", "thiotimoline.com",
    "thisisnotmyrealemail.com", "thisisvalid.com",
    "thomasandpeters.com", "thottbot.com",
    "throwam.com", "throwawayemailaddress.com",
    "throwawaymail.com", "throam.com",
    "thunk.com", "thundermail.net",
    "tiem.hu", "tiffanysoccer.com",
    "tikitruck.com", "timacadabra.fr",
    "timewarner.net", "tin.it",
    "tinyurl24.com", "titanemail.com",
    "tk-garage.com", "tlen.pl",
    "tm.tc", "tmpbe.com",
    "tmpeml.net", "tmpeml.org",
    "tmprss.com", "tmpwrld.com",
    "tnhl.com", "toiee.com",
    "tokem.co", "tokenmail.de",
    "tonyonline.de", "toomail.biz",
    "top100.de", "topmail1.com",
    "topmail2.com", "topmail3.com",
    "topmail4.com", "topmail5.com",
    "topmail6.com", "topmail7.com",
    "topmail8.com", "topmail9.com",
    "topsitelinks.com", "torreviejacomercio.es",
    "toss.pw", "totalise.co.uk",
    "totallyuseless.com", "towyardcars.com",
    "toyota.com.nyud.net", "tr22.net",
    "tr7mail.com", "traf.dk",
    "trash-amil.com", "trash-mail.at",
    "trash-mail.cf", "trash-mail.de",
    "trash-mail.ga", "trash-mail.gq",
    "trash-mail.ml", "trash-mail.tk",
    "trash2009.com", "trashcanmail.com",
    "trashdevil.com", "trashemails.de",
    "trashinbox.com", "trashmail.at",
    "trashmail.com.br", "trashmail.de",
    "trashmail.gr", "trashmail.io",
    "trashmail.me", "trashmail.net",
    "trashmail.org", "trashmail.ws",
    "trashmailer.com", "trashmailz.de",
    "trashymail.com", "trashymail.net",
    "trbvm.com", "trean.org",
    "trickmail.net", "trillville.net",
    "triplehelix6.com", "trinitycap.net",
    "trixan.com", "trpmail.com",
    "truck2handover.com", "truechristianfaith.com",
    "trumail.net", "trust-7.com",
    "tsu.edu", "tttn.de",
    "tupi.com.br", "turboninja.cz",
    "turok.org", "tvchd.com",
    "twinmail.de", "twkly.com",
    "twogroceries.com", "txtadvertise.com",
    "ty.gy", "tyhe.net",
    "uacro.com", "uboot.com",
    "ubismail.net", "uconnect.at",
    "uol.com.nyud.net", "upliftnow.com",
    "upload-8.com", "upliftingheartsradio.com",
    "upoznaj.to", "url.com.uy",
    "urgent365.com", "urxtech.com",
    "usa.com", "used-product.com",
    "users.sourceforge.net.nyud.net", "ushijima.info",
    "usmalldata.com", "uspto.gov.nyud.net",
    "usinglinux.net", "usinternet.com",
    "utoo.pw", "uu.gl",
    "uwork4.us", "ux24.biz",
    "ux3.com", "uzayportali.com",
    "v33x.com", "vanderlinden.us",
    "vantronix.net", "vaporobject.com",
    "vbulletin.com.nyud.net", "vcn.com",
    "vegan4life.info", "venompen.com",
    "verizon.net", "veryfast.biz",
    "vfemail.net", "vi2x.com",
    "vicecrew.com", "victim13.com",
    "video-email.info", "vidscaling.com",
    "viktorfarfar.com", "vinternet.com",
    "vipepe.com", "virtual-mail.nl",
    "virtual-mailbox.org", "virgilio.it",
    "virus.org", "visymail.com",
    "vixlet.com", "vlug.ug",
    "vmailing.info", "vmpf.co.uk",
    "vodkacat.com", "voila.fr",
    "vomoto.com", "voodoo.net",
    "voprea.com", "vote2016.co.uk",
    "vpnsecure.me", "vrmtr.com",
    "vtext.com.nyud.net", "vvv5.com",
    "vzwpix.com.nyud.net", "w3internet.co.uk",
    "w3s.com", "w7xp.net",
    "walla.co.il", "wallawallaflooring.com",
    "walter-service.de", "wamu.com.nyud.net",
    "wanadoo.com.nyud.net", "wankersrus.com",
    "warpmail.net", "washington.edu",
    "waterok.org", "wbcollins.net",
    "wearacardigan.com", "web2mail.com",
    "webemail.me", "webhome24.com",
    "webideal.com", "webinz.ro",
    "webm4il.info", "webmail2000.com",
    "websolute.be", "webtopmail.com",
    "weedfarmer.org", "weekaw.com",
    "weightx.net", "weldnertel.com",
    "wemeetmag.com", "westnet.com.au",
    "wetalkchicks.com", "wfec2010.com",
    "whatifanalytics.com", "whatpaas.com",
    "whatsthebeef.net", "wh4u.de",
    "whipmail.net", "whsmith.co.uk",
    "whyspam.me", "wibble.net",
    "wickmail.net", "wielder.com",
    "wifitrack.de", "wika-szczepanski.de",
    "wikiwix.com", "wildmail.co.uk",
    "williamgrossett.com", "willie2009.com",
    "willreport.com", "wilma.it",
    "wim.mine.nu", "windstream.net",
    "winob.com", "wintelguy.com",
    "winx6.com", "wisconsin.gov.nyud.net",
    "wismail.com", "wmailer.com",
    "wmpoweruser.com", "wosu.com",
    "work.com", "workmail.info",
    "worldspace.eu", "wosx.net",
    "wow.com", "wowappmail.com",
    "wpgadv.com", "wptemplates.us",
    "writeme.com", "wronghead.com",
    "wso2.com", "wuaze.com",
    "www.newmail.ru", "wxnw.net",
    "wz66.com", "x24.com",
    "x7rqw.com", "xaa.ath.cx",
    "xcode.com", "xemaps.com",
    "xfront.com", "xgets.com",
    "xiglute.com", "xinu.at",
    "xiongmai.com", "xl.ph",
    "xmaily.com", "xmail.com",
    "xmsnet.com", "xobdoj.com",
    "xooit.com", "xorg.za.net",
    "xperiaeiro.com", "xproxy.com",
    "xrated.biz", "xtra.co.nz",
    "xwaretech.com", "xxhamsterxx.com",
    "xxxclubporno.com", "xxxmail.de",
    "xxxsexyporn.com", "xz.am",
    "y7u89.com", "yandex.com.nyud.net",
    "yandex.ru", "yawn.net",
    "ycn.ro", "ydesk.net",
    "yellowgorilla.net", "yen.com.gh",
    "yesey.net", "yggmail.com",
    "yogamaven.com", "yola.net",
    "you-spam.com", "yougotgoated.com",
    "yourlms.com", "youwannabet.com",
    "yopmail.fr", "yopmail.com",
    "yopmail.net", "yopmail.org",
    "yortm.org", "youmail.com",
    "yourdomain.com", "yourleisure.info",
    "ypmail.web.id", "yuurok.com",
    "z1p.biz", "zabbo.com",
    "zackyfamily.com", "zain.site",
    "zaktouni.fr", "zapis.to",
    "zb8.com", "zdemail.com",
    "zedtek.com", "zee-email.com",
    "zemni.com", "zen-email.com",
    "zencash.com", "zeroentry.net",
    "zetmail.com", "zhew.com",
    "zik.dj", "zipcon.net",
    "zoaxe.com", "zoemail.com",
    "zoemail.net", "zombie-hive.com",
    "zomg.info", "zpost.pl",
    "zumpul.com", "zvibes.com",
    "zydecodriver.com", "zyx5.com",
}


DISPOSABLE_EMAIL_DOMAINS_PATTERNS = [
    re.compile(r'^temp(?:mail)?[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^dispos(?:able)?[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^throw(?:away)?[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^trash(?:mail)?[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^fake[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^anonymous[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^no[.-]?spam[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^spam[.-]?(?:free|block|grab|catch|hole|away|off)\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^10(?:minutemail|minutes?mail)[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^60(?:minutemail|minutes?mail)[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^24(?:hourmail|hours?mail)[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^guerrilla[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^yopmail[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^sharklasers[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^mailinator[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^getnada[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^mint[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
    re.compile(r'^moakt[.-]?\w*\.(?:com|net|org|info|biz|ru|de|fr|uk|us|eu|io|co|to|tk|ml|ga|gq|cf)$', re.IGNORECASE),
]


@dataclass
class ValidationResult:
    email: str
    format_valid: bool = False
    format_errors: list = field(default_factory=list)
    mx_exists: bool = False
    mx_records: list = field(default_factory=list)
    mx_cached: bool = False
    is_disposable: bool = False
    smtp_verified: bool = False
    smtp_success: bool = False
    smtp_message: str = ""
    smtp_cached: bool = False
    suggestions: list = field(default_factory=list)

    @property
    def is_deliverable(self) -> bool:
        return self.format_valid and self.mx_exists and not self.is_disposable

    @property
    def is_fully_verified(self) -> bool:
        return self.is_deliverable and self.smtp_verified and self.smtp_success


COMMON_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "qq.com", "163.com", "126.com", "sina.com", "foxmail.com",
    "icloud.com", "protonmail.com", "mail.com", "yandex.com",
}

DOMAIN_TYPOS = {
    "gmial.com": "gmail.com",
    "gamil.com": "gmail.com",
    "gmai.com": "gmail.com",
    "gmail.co": "gmail.com",
    "gmal.com": "gmail.com",
    "yaho.com": "yahoo.com",
    "yahooo.com": "yahoo.com",
    "yahho.com": "yahoo.com",
    "hotmal.com": "hotmail.com",
    "hotmial.com": "hotmail.com",
    "hotmai.com": "hotmail.com",
    "outlok.com": "outlook.com",
    "outlook.co": "outlook.com",
    "qq.co": "qq.com",
    "163.co": "163.com",
    "126.co": "126.com",
}

EMAIL_REGEX = re.compile(
    r"^(?P<local>[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+)"
    r"@"
    r"(?P<domain>(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})$"
)

LOCAL_PART_STRICT_REGEX = re.compile(
    r"^[a-zA-Z0-9](?:[a-zA-Z0-9._+-]*[a-zA-Z0-9])?$"
)


def is_disposable_email(domain: str) -> bool:
    if not domain:
        return False
    domain_lower = domain.lower()
    if domain_lower in DISPOSABLE_EMAIL_DOMAINS:
        return True
    for pattern in DISPOSABLE_EMAIL_DOMAINS_PATTERNS:
        if pattern.match(domain_lower):
            return True
    return False


def get_mx_cache(domain: str) -> Optional[tuple[bool, list[str]]]:
    entry = MX_CACHE.get(domain.lower())
    if entry and time.time() - entry[0] < MX_CACHE_TTL:
        return (entry[1], entry[2])
    if entry:
        del MX_CACHE[domain.lower()]
    return None


def set_mx_cache(domain: str, mx_exists: bool, mx_records: list[str]) -> None:
    MX_CACHE[domain.lower()] = (time.time(), mx_exists, mx_records)


def get_smtp_cache(email: str) -> Optional[tuple[bool, str]]:
    entry = SMTP_CACHE.get(email.lower())
    if entry and time.time() - entry[0] < SMTP_CACHE_TTL:
        return (entry[1], entry[2])
    if entry:
        del SMTP_CACHE[email.lower()]
    return None


def set_smtp_cache(email: str, success: bool, message: str) -> None:
    SMTP_CACHE[email.lower()] = (time.time(), success, message)


def clear_expired_cache() -> None:
    now = time.time()
    expired_mx = [d for d, (t, _, _) in MX_CACHE.items() if now - t >= MX_CACHE_TTL]
    for d in expired_mx:
        del MX_CACHE[d]
    expired_smtp = [e for e, (t, _, _) in SMTP_CACHE.items() if now - t >= SMTP_CACHE_TTL]
    for e in expired_smtp:
        del SMTP_CACHE[e]


def extract_mx_host(mx_record: str) -> str:
    parts = mx_record.split()
    if len(parts) >= 2:
        return parts[1]
    return mx_record


def validate_format(email: str) -> tuple[bool, list[str], str, str]:
    errors = []

    if not email or not isinstance(email, str):
        return False, ["邮箱地址不能为空"], "", ""

    email = email.strip()

    if " " in email:
        errors.append("邮箱地址中不能包含空格")

    if email.count("@") == 0:
        errors.append("缺少 @ 符号")
        return False, errors, "", ""
    elif email.count("@") > 1:
        errors.append("邮箱地址中只能包含一个 @ 符号")
        return False, errors, "", ""

    local_part, domain = email.rsplit("@", 1)

    if not local_part:
        errors.append("@ 前的用户名部分不能为空")
    elif len(local_part) > 64:
        errors.append("用户名部分长度不能超过64个字符")
    else:
        if local_part.startswith("."):
            errors.append("用户名不能以点号开头")
        if local_part.endswith("."):
            errors.append("用户名不能以点号结尾")
        if ".." in local_part:
            errors.append("用户名中不能包含连续的点号")
        if not LOCAL_PART_STRICT_REGEX.match(local_part):
            errors.append("用户名包含不合法的字符（仅允许字母、数字及 . _ + -）")

    if not domain:
        errors.append("@ 后的域名部分不能为空")
    elif len(domain) > 255:
        errors.append("域名长度不能超过255个字符")
    else:
        if domain.startswith("-") or domain.endswith("-"):
            errors.append("域名不能以连字符开头或结尾")
        if domain.startswith(".") or domain.endswith("."):
            errors.append("域名不能以点号开头或结尾")

    match = EMAIL_REGEX.match(email)
    if not match and not errors:
        errors.append("邮箱格式不合法")

    is_valid = match is not None and len(errors) == 0
    local_out = match.group("local") if match else local_part
    domain_out = match.group("domain") if match else domain

    return is_valid, errors, local_out, domain_out


def check_mx_records(domain: str, timeout: float = DNS_TIMEOUT) -> tuple[bool, list[str], bool]:
    cached = get_mx_cache(domain)
    if cached is not None:
        return cached[0], cached[1], True

    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout
        answers = resolver.resolve(domain, "MX")
        records = sorted(
            [(r.preference, str(r.exchange).rstrip(".")) for r in answers],
            key=lambda x: x[0],
        )
        mx_records = [f"{pref} {exch}" for pref, exch in records]
        set_mx_cache(domain, True, mx_records)
        return True, mx_records, False
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
            dns.resolver.NoNameservers, dns.exception.Timeout):
        set_mx_cache(domain, False, [])
        return False, [], False
    except Exception:
        set_mx_cache(domain, False, [])
        return False, [], False


async def check_mx_records_async(
    domain: str,
    timeout: float = DNS_TIMEOUT,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> tuple[bool, list[str], bool]:
    cached = get_mx_cache(domain)
    if cached is not None:
        return cached[0], cached[1], True

    if semaphore is not None:
        async with semaphore:
            return await _do_async_mx_query(domain, timeout)
    return await _do_async_mx_query(domain, timeout)


async def _do_async_mx_query(domain: str, timeout: float) -> tuple[bool, list[str], bool]:
    try:
        resolver = dns.asyncresolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout
        answers = await resolver.resolve(domain, "MX")
        records = sorted(
            [(r.preference, str(r.exchange).rstrip(".")) for r in answers],
            key=lambda x: x[0],
        )
        mx_records = [f"{pref} {exch}" for pref, exch in records]
        set_mx_cache(domain, True, mx_records)
        return True, mx_records, False
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
            dns.resolver.NoNameservers, dns.exception.Timeout):
        set_mx_cache(domain, False, [])
        return False, [], False
    except Exception:
        set_mx_cache(domain, False, [])
        return False, [], False


def check_smtp_handshake(
    email: str,
    mx_records: list[str],
    timeout: float = SMTP_TIMEOUT,
    sender_email: str = "verify@example.com",
) -> tuple[bool, bool, str]:
    if not mx_records:
        return False, False, "无MX记录"

    cached = get_smtp_cache(email)
    if cached is not None:
        return True, cached[0], cached[1]

    smtp = None
    try:
        for mx_record in mx_records:
            mx_host = extract_mx_host(mx_record)
            try:
                smtp = smtplib.SMTP(timeout=timeout)
                smtp.connect(mx_host, 25)
                smtp.ehlo()

                try:
                    smtp.starttls()
                    smtp.ehlo()
                except Exception:
                    pass

                code, message = smtp.mail(sender_email)
                if code != 250:
                    continue

                code, message = smtp.rcpt(email)
                msg_str = message.decode('utf-8', errors='replace') if isinstance(message, bytes) else str(message)

                if code == 250:
                    set_smtp_cache(email, True, msg_str)
                    try:
                        smtp.quit()
                    except Exception:
                        pass
                    return True, True, msg_str
                elif code in (550, 551, 552, 553, 554):
                    set_smtp_cache(email, False, msg_str)
                    try:
                        smtp.quit()
                    except Exception:
                        pass
                    return True, False, f"邮箱不存在: {msg_str}"
                else:
                    continue

            except (socket.timeout, socket.error, ConnectionRefusedError, OSError):
                continue
            except smtplib.SMTPException:
                continue
            finally:
                if smtp is not None:
                    try:
                        smtp.close()
                    except Exception:
                        pass
                smtp = None

        return False, False, "无法连接到邮件服务器或验证失败"
    except Exception as e:
        msg = str(e)
        set_smtp_cache(email, False, msg)
        return False, False, msg


def generate_suggestions(
    email: str, local_part: str, domain: str,
    format_valid: bool, mx_exists: bool, mx_records: list[str],
    is_disposable: bool,
    smtp_verified: bool, smtp_success: bool, smtp_message: str,
) -> list[str]:
    suggestions = []

    if not format_valid:
        suggestions.append("请检查邮箱地址格式是否正确")

    lower_domain = domain.lower()
    if lower_domain in DOMAIN_TYPOS:
        corrected = DOMAIN_TYPOS[lower_domain]
        suggested_email = f"{local_part}@{corrected}"
        suggestions.append(
            f"域名 '{lower_domain}' 可能是拼写错误，是否指 '{corrected}'？建议邮箱：{suggested_email}"
        )

    if is_disposable:
        suggestions.append(
            f"⚠️  检测到 '{domain}' 为一次性/临时邮箱域名，不建议用于正式业务"
        )

    if format_valid and not is_disposable and not mx_exists:
        suggestions.append(
            f"域名 '{domain}' 未找到有效的MX记录，该邮箱可能无法接收邮件"
        )
        if lower_domain not in COMMON_DOMAINS:
            suggestions.append(
                "常见有效邮箱域名：gmail.com、outlook.com、qq.com、163.com 等，请确认域名拼写"
            )
        suggestions.append(f"域名 '{domain}' 可能不存在或尚未配置邮件服务")

    if format_valid and mx_exists and not is_disposable:
        if len(mx_records) == 1:
            suggestions.append("该域名仅有一条MX记录，存在单点故障风险")

        if smtp_verified:
            if smtp_success:
                suggestions.append("✅ SMTP握手验证通过，该邮箱真实存在且可以接收邮件")
            else:
                suggestions.append(f"❌ SMTP验证失败：{smtp_message}")
        else:
            suggestions.append("MX记录验证通过，邮箱大概率可以接收邮件（但无法确认具体用户是否存在，可启用SMTP深度验证）")

    return suggestions


def validate_email(email: str, check_smtp: bool = False) -> ValidationResult:
    result = ValidationResult(email=email.strip())

    format_valid, errors, local_part, domain = validate_format(email)
    result.format_valid = format_valid
    result.format_errors = errors

    if format_valid and domain:
        result.is_disposable = is_disposable_email(domain)

        if not result.is_disposable:
            mx_exists, mx_records, mx_cached = check_mx_records(domain)
            result.mx_exists = mx_exists
            result.mx_records = mx_records
            result.mx_cached = mx_cached

            if check_smtp and mx_exists and mx_records:
                smtp_verified, smtp_success, smtp_message = check_smtp_handshake(email, mx_records)
                result.smtp_verified = smtp_verified
                result.smtp_success = smtp_success
                result.smtp_message = smtp_message
                result.smtp_cached = smtp_verified and get_smtp_cache(email) is not None
        else:
            result.mx_exists = False
            result.mx_records = []
            result.mx_cached = False
    else:
        result.mx_exists = False
        result.mx_records = []
        result.mx_cached = False

    result.suggestions = generate_suggestions(
        email, local_part, domain,
        format_valid, result.mx_exists, result.mx_records,
        result.is_disposable,
        result.smtp_verified, result.smtp_success, result.smtp_message,
    )

    return result


async def validate_email_async(
    email: str,
    check_smtp: bool = False,
    dns_semaphore: Optional[asyncio.Semaphore] = None,
    smtp_semaphore: Optional[asyncio.Semaphore] = None,
) -> ValidationResult:
    result = ValidationResult(email=email.strip())

    format_valid, errors, local_part, domain = validate_format(email)
    result.format_valid = format_valid
    result.format_errors = errors

    if format_valid and domain:
        result.is_disposable = is_disposable_email(domain)

        if not result.is_disposable:
            mx_exists, mx_records, mx_cached = await check_mx_records_async(
                domain, semaphore=dns_semaphore
            )
            result.mx_exists = mx_exists
            result.mx_records = mx_records
            result.mx_cached = mx_cached

            if check_smtp and mx_exists and mx_records:
                smtp_verified, smtp_success, smtp_message = await check_smtp_handshake_async(
                    email, mx_records, semaphore=smtp_semaphore
                )
                result.smtp_verified = smtp_verified
                result.smtp_success = smtp_success
                result.smtp_message = smtp_message
                result.smtp_cached = smtp_verified and get_smtp_cache(email) is not None
        else:
            result.mx_exists = False
            result.mx_records = []
            result.mx_cached = False
    else:
        result.mx_exists = False
        result.mx_records = []
        result.mx_cached = False

    result.suggestions = generate_suggestions(
        email, local_part, domain,
        format_valid, result.mx_exists, result.mx_records,
        result.is_disposable,
        result.smtp_verified, result.smtp_success, result.smtp_message,
    )

    return result


async def validate_emails_batch(
    emails: list[str],
    check_smtp: bool = False,
    progress_callback: Optional[Callable[[int, int, ValidationResult], None]] = None,
    max_concurrent_dns: int = MAX_CONCURRENT_DNS,
    max_concurrent_smtp: int = MAX_CONCURRENT_SMTP,
) -> list[ValidationResult]:
    dns_semaphore = asyncio.Semaphore(max_concurrent_dns)
    smtp_semaphore = asyncio.Semaphore(max_concurrent_smtp) if check_smtp else None

    tasks = [
        validate_email_async(email, check_smtp, dns_semaphore, smtp_semaphore)
        for email in emails
    ]

    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        if progress_callback:
            progress_callback(i, len(emails), result)

    return results


def validate_emails_batch_sync(
    emails: list[str],
    check_smtp: bool = False,
    progress_callback: Optional[Callable[[int, int, ValidationResult], None]] = None,
    max_concurrent_dns: int = MAX_CONCURRENT_DNS,
    max_concurrent_smtp: int = MAX_CONCURRENT_SMTP,
) -> list[ValidationResult]:
    return asyncio.run(validate_emails_batch(
        emails, check_smtp, progress_callback,
        max_concurrent_dns, max_concurrent_smtp
    ))


def format_result(result: ValidationResult) -> str:
    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"邮箱验证结果: {result.email}")
    lines.append(f"{'='*50}")
    lines.append(f"  格式校验: {'✅ 通过' if result.format_valid else '❌ 不通过'}")

    if result.format_errors:
        lines.append(f"  格式错误:")
        for err in result.format_errors:
            lines.append(f"    - {err}")

    lines.append(f"  一次性邮箱: {'⚠️  是' if result.is_disposable else '✅ 否'}")

    if result.format_valid:
        cache_tag = " (缓存)" if result.mx_cached else ""
        lines.append(f"  MX记录:   {'✅ 存在' if result.mx_exists else '❌ 不存在'}{cache_tag}")
        if result.mx_records:
            lines.append(f"  MX记录详情:")
            for rec in result.mx_records:
                lines.append(f"    - {rec}")

        if result.smtp_verified:
            smtp_cache = " (缓存)" if result.smtp_cached else ""
            lines.append(f"  SMTP验证: {'✅ 通过' if result.smtp_success else '❌ 失败'}{smtp_cache}")
            if result.smtp_message:
                lines.append(f"  SMTP详情: {result.smtp_message[:80]}")

    status = "✅ 可投递" if result.is_deliverable else "⚠️  不可投递"
    if result.is_fully_verified:
        status = "✅✅ 完全验证"
    lines.append(f"  可投递性: {status}")

    if result.suggestions:
        lines.append(f"  建议:")
        for sug in result.suggestions:
            lines.append(f"    💡 {sug}")

    lines.append(f"{'='*50}")
    return "\n".join(lines)


def format_batch_summary(results: list[ValidationResult], elapsed: float) -> str:
    total = len(results)
    valid_format = sum(1 for r in results if r.format_valid)
    disposable = sum(1 for r in results if r.is_disposable)
    deliverable = sum(1 for r in results if r.is_deliverable)
    fully_verified = sum(1 for r in results if r.is_fully_verified)
    mx_cached = sum(1 for r in results if r.mx_cached)
    smtp_cached = sum(1 for r in results if r.smtp_cached)

    lines = []
    lines.append("\n" + "="*60)
    lines.append("批量验证汇总")
    lines.append("="*60)
    lines.append(f"  总邮箱数:    {total}")
    lines.append(f"  格式有效:    {valid_format}/{total} ({valid_format/total*100:.1f}%)")
    lines.append(f"  一次性邮箱:  {disposable}/{total} ({disposable/total*100:.1f}%)")
    lines.append(f"  可投递:      {deliverable}/{total} ({deliverable/total*100:.1f}%)")
    lines.append(f"  完全验证:    {fully_verified}/{total} ({fully_verified/total*100:.1f}%)")
    lines.append(f"  MX缓存命中:  {mx_cached} 次")
    if smtp_cached > 0:
        lines.append(f"  SMTP缓存:    {smtp_cached} 次")
    lines.append(f"  总耗时:      {elapsed:.2f} 秒")
    lines.append(f"  平均每个:    {elapsed/total*1000:.1f} 毫秒")
    lines.append("="*60)
    return "\n".join(lines)


if __name__ == "__main__":
    test_emails = [
        "user@gmail.com",
        "test@qq.com",
        "invalid-email",
        "user@nonexistentdomain12345.com",
        "user@gmial.com",
        "user.name+tag@outlook.com",
        "@example.com",
        "user@",
        "user@.com",
        "user..dot@gmail.com",
        "test@temp-mail.org",
        "user@yopmail.com",
        "test@10minutemail.com",
    ]

    print("="*60)
    print("测试1: 单个验证 (含一次性邮箱检测)")
    print("="*60)

    for addr in test_emails[:5] + test_emails[-3:]:
        result = validate_email(addr)
        print(format_result(result))
        print()

    print("\n" + "="*60)
    print("测试2: 批量异步验证 (重复域名将使用缓存)")
    print("="*60)

    batch_emails = test_emails + ["test2@gmail.com", "admin@qq.com", "user3@gmail.com"]

    start = time.time()

    def progress(current, total, result):
        status = "✅" if result.is_deliverable else "❌"
        disp = " [一次性]" if result.is_disposable else ""
        print(f"  [{current}/{total}] {status} {result.email}{disp}")

    results = validate_emails_batch_sync(batch_emails, progress_callback=progress)
    elapsed = time.time() - start

    print(format_batch_summary(results, elapsed))

    print("\n" + "="*60)
    print("测试3: 缓存验证 - 再次批量查询相同域名 (应该更快)")
    print("="*60)

    start = time.time()
    results2 = validate_emails_batch_sync(batch_emails)
    elapsed2 = time.time() - start
    print(format_batch_summary(results2, elapsed2))

    print("\n" + "="*60)
    print("测试4: SMTP深度验证 (注意：可能较慢且部分服务器会拒绝连接)")
    print("="*60)

    smtp_test_emails = [
        "noreply@github.com",
        "invalid-user-12345@gmail.com",
    ]

    for addr in smtp_test_emails:
        print(f"\n正在验证 {addr} ...")
        start = time.time()
        result = validate_email(addr, check_smtp=True)
        elapsed_smtp = time.time() - start
        print(format_result(result))
        print(f"  耗时: {elapsed_smtp:.2f} 秒")