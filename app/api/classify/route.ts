import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(req: Request) {
  try {
    const { text } = await req.json();

    if (!text) {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    // Path to your python script
    const pythonScript = path.join(process.cwd(), 'scripts', 'classify_json.py');
    
    // Spawn the python process
    // Note: You might need to use 'python3' instead of 'python' on some systems.
    // On Windows, 'python' or 'py' is common.
    const pythonProcess = spawn('python', [pythonScript]);

    let output = '';
    let errorOutput = '';

    // Send input to stdin
    pythonProcess.stdin.write(text);
    pythonProcess.stdin.end();

    // Capture stdout and stderr
    for await (const chunk of pythonProcess.stdout) {
      output += chunk.toString();
    }
    for await (const chunk of pythonProcess.stderr) {
      errorOutput += chunk.toString();
    }

    // Wait for the process to finish
    await new Promise((resolve) => pythonProcess.on('close', resolve));

    if (errorOutput && !output) {
      console.error('Python Error:', errorOutput);
      return NextResponse.json({ error: `Internal Server Error: ${errorOutput}` }, { status: 500 });
    }

    try {
      // Find the JSON part in the output (in case of extra print statements)
      const jsonStart = output.indexOf('{');
      const jsonEnd = output.lastIndexOf('}') + 1;
      const jsonStr = output.slice(jsonStart, jsonEnd);
      
      const result = JSON.parse(jsonStr);
      
      if (result.error) {
        return NextResponse.json({ error: result.error }, { status: 400 });
      }
      
      return NextResponse.json(result);
    } catch (parseError) {
      console.error('Parse Error:', parseError, 'Raw Output:', output);
      return NextResponse.json({ error: 'Failed to parse model output', raw: output }, { status: 500 });
    }

  } catch (error: any) {
    console.error('API Error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
