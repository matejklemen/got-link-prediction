const ky = require('ky-universal');
const range = require('range');

const getKiller = killString => {
    // get everything after the word by in the kill string
    // e.g.: Killed by Khal Drogo => Khal Drogo
    // returns null if there is no "by" in the string

    const beforeAndAfterBy = killString.split("by ");

    if (beforeAndAfterBy.length > 1) {
        // returns the element after the last "by" word in the string
        return killString.split("by ").pop();
    } else {
        return null;
    }
}


(async () => {
    // Create csv header
    const header = "Character, Killed by, Time, House, Status, Importance, Episode ID";
    console.log(header);

    const seasons = range.range(1, 7);
    seasons.forEach(async season => {
        try {
            const deaths = await ky.get(`https://deathtimeline.com/api/deaths?season=${season}`).json();
            deaths.forEach(death => {
                const row = [death.name, getKiller(death.killedBy), death.time, death.house, death.status, death.importance, death.episode_id]
                    // adds a string undefined if the value is not defined instead of leaving value empty in the string
                    .map(value => value || "undefined")
                    // joins the values into a comma-separated string
                    .join(", ");

                console.log(row);
            });
        } catch (e) {
            console.log("Season not found on API.");
            console.log(e.stack);
        }
    })
})();
