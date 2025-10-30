// "use client"

// import { useState, useEffect } from "react"
// import { Button } from "../../components/ui/button"
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
// import { Input } from "../../components/ui/input"
// import { Textarea } from "../../components/ui/textarea"

// const CBT_PROMPTS = [
//   "What thoughts are you having right now?",
//   "What emotions are you experiencing?",
//   "What triggered these feelings?",
//   "How can you reframe this situation?",
// ]

// export default function JournalModule({ userId }: { userId: string }) {
//   const [title, setTitle] = useState("")
//   const [content, setContent] = useState("")
//   const [isLoading, setIsLoading] = useState(false)
//   const [journals, setJournals] = useState<any[]>([])
//   const [message, setMessage] = useState("")
//   const [selectedPrompt, setSelectedPrompt] = useState("")

//   useEffect(() => {
//     fetchJournals()
//   }, [userId])

//   const fetchJournals = async () => {
//     try {
//       const token = localStorage.getItem("token")
//       const res = await fetch(`/api/journal?userId=${userId}`, {
//         headers: { Authorization: `Bearer ${token}` },
//       })
//       if (res.ok) {
//         const data = await res.json()
//         setJournals(data)
//       }
//     } catch (error) {
//       console.error("Failed to fetch journals:", error)
//     }
//   }

//   const handleSubmit = async () => {
//     if (!title || !content) {
//       setMessage("Please fill in all fields")
//       return
//     }

//     setIsLoading(true)
//     setMessage("")

//     try {
//       const token = localStorage.getItem("token")
//       const res = await fetch("/api/journal", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           Authorization: `Bearer ${token}`,
//         },
//         body: JSON.stringify({
//           userId,
//           title,
//           content,
//         }),
//       })

//       if (res.ok) {
//         setMessage("Journal entry saved!")
//         setTitle("")
//         setContent("")
//         setSelectedPrompt("")
//         fetchJournals()
//       } else {
//         setMessage("Failed to save entry")
//       }
//     } catch (error) {
//       setMessage("An error occurred")
//     } finally {
//       setIsLoading(false)
//     }
//   }

//   return (
//     <div className="space-y-6">
//       <Card>
//         <CardHeader>
//           <CardTitle>Write Your Journal</CardTitle>
//           <CardDescription>Reflect on your thoughts and feelings with guided prompts</CardDescription>
//         </CardHeader>
//         <CardContent className="space-y-4">
//           <div className="space-y-2">
//             <label className="text-sm font-medium">Guided Prompts</label>
//             <div className="grid grid-cols-2 gap-2">
//               {CBT_PROMPTS.map((prompt) => (
//                 <button
//                   key={prompt}
//                   onClick={() => setSelectedPrompt(prompt)}
//                   className={`p-3 text-left text-sm rounded-lg transition-all ${
//                     selectedPrompt === prompt ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
//                   }`}
//                 >
//                   {prompt}
//                 </button>
//               ))}
//             </div>
//           </div>

//           <div className="space-y-2">
//             <label className="text-sm font-medium">Title</label>
//             <Input placeholder="Give your entry a title" value={title} onChange={(e) => setTitle(e.target.value)} />
//           </div>

//           <div className="space-y-2">
//             <label className="text-sm font-medium">Your Thoughts {selectedPrompt && `- ${selectedPrompt}`}</label>
//             <Textarea
//               placeholder="Write freely about your thoughts and feelings..."
//               value={content}
//               onChange={(e) => setContent(e.target.value)}
//               className="min-h-32"
//             />
//           </div>

//           {message && (
//             <div
//               className={`p-3 rounded-md text-sm ${
//                 message.includes("saved") ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
//               }`}
//             >
//               {message}
//             </div>
//           )}

//           <Button onClick={handleSubmit} disabled={isLoading} className="w-full">
//             {isLoading ? "Saving..." : "Save Entry"}
//           </Button>
//         </CardContent>
//       </Card>

//       {journals.length > 0 && (
//         <Card>
//           <CardHeader>
//             <CardTitle>Your Journal Entries</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <div className="space-y-3">
//               {journals.slice(0, 5).map((journal: any) => (
//                 <div key={journal._id} className="p-4 bg-muted rounded-lg">
//                   <h3 className="font-semibold">{journal.title}</h3>
//                   <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{journal.content}</p>
//                   <p className="text-xs text-muted-foreground mt-2">
//                     {new Date(journal.createdAt).toLocaleDateString()}
//                   </p>
//                 </div>
//               ))}
//             </div>
//           </CardContent>
//         </Card>
//       )}
//     </div>
//   )
// }
