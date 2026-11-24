import Foundation

@main
enum SnowfallOutlookTests {
    static func main() {
        let origin = parseDate("2026-11-23T06:00:00Z")!
        let clock = ISO8601DateFormatter()
        func stamp(_ hour: Int) -> String {
            clock.string(from: origin.addingTimeInterval(Double(hour) * 3600))
        }
        func readings(_ cm: [Double], firstHour: Int = 0) -> [[String: Any]] {
            cm.enumerated().map { slot, amount in
                ["start": stamp(firstHour + slot), "time": stamp(firstHour + slot + 1), "snowfall_inches": amount / 2.54]
            }
        }
        let dashboard = readings([0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.49, 0.49])
        let comparison = readings([0.28, 0.28, 0.28, 0.49, 0, 0, 0, 0], firstHour: 1)
        let record: [String: Any] = ["quantity": "snowfall", "status": "ready", "mean": 1.57,
                                    "start": stamp(1), "end": stamp(9), "provider_mean": 1.33 / 2.54,
                                    "provider_hourly": comparison]
        func outlook(_ fixture: [String: Any]) -> SnowfallOutlook {
            SnowfallOutlook(payload: ["hourly": dashboard], record: fixture, hours: 8, now: origin)
        }
        let matched = outlook(record)
        precondition(matched.paired && matched.start == origin.addingTimeInterval(3600))
        precondition(abs(matched.total! * 2.54 - 1.33) < 1e-9)
        precondition(matched.amounts[0] == 0.28 / 2.54 && matched.amounts[7] == 0)
        var fixture = record
        fixture["provider_hourly"] = Array(comparison.dropLast()) + [comparison[0]]
        precondition(outlook(fixture).total == nil)
        fixture = record
        fixture["provider_mean"] = NSNull()
        precondition(outlook(fixture).total == nil)
        fixture = record
        fixture["provider_mean"] = 0.0
        fixture["provider_hourly"] = readings(Array(repeating: 0, count: 8), firstHour: 1)
        precondition(outlook(fixture).total == 0)
        fixture = record
        fixture["quantity"] = "liquid_precipitation"
        let liquid = outlook(fixture)
        precondition(!liquid.paired && abs(liquid.total! * 2.54 - 2.66) < 1e-9)
        fixture = record
        fixture.removeValue(forKey: "provider_hourly")
        precondition(!outlook(fixture).paired && abs(outlook(fixture).total! * 2.54 - 2.66) < 1e-9)
        print("Passed snowfall window regression checks")
    }
}
