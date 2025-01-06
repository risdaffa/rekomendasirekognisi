import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Lightbulb, ChevronRight } from "lucide-react";

export function RecommendationCard({ recommendation }) {
  return (
    <Card className="mt-8 shadow-md overflow-hidden">
      <CardHeader className="bg-gray-100 dark:bg-gray-800">
        <CardTitle className="text-xl font-bold flex items-center text-gray-800 dark:text-gray-200">
          <Lightbulb className="mr-2" />
          Course Recommendations
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        {recommendation.recommendation.map((rec, index) => (
          <div key={index} className="mb-8 last:mb-0">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
              {rec.dataSI}
            </h3>
            <div className="space-y-4">
              {rec.recommendations.map((course, idx) => (
                <div
                  key={idx}
                  className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 transition-all hover:shadow-md"
                >
                  <div className="flex items-center mb-2">
                    <ChevronRight className="mr-2 h-4 w-4 text-blue-500" />
                    <h4 className="font-medium text-gray-900 dark:text-gray-100">
                      {course["MK Rekomendasi"]}
                    </h4>
                  </div>
                  <div className="grid gap-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">
                        P(C|S) = Persentase {rec.dataSI} yang dimuat pada {course["MK Rekomendasi"]}
                        </span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {(course["P(B|A)"] * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={course["P(B|A)"] * 100}
                        className="h-2"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">
                        P(S|C) = Persentase {course["MK Rekomendasi"]} yang dimuat pada {rec.dataSI}
                        </span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {(course["P(A|B)"] * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={course["P(A|B)"] * 100}
                        className="h-2"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
